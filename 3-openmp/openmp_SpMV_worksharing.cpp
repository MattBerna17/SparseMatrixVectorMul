// OpenMP WORKSHARING implementation of the Iterative Sparse Matrix-Vector Computation on Evolving Sparse Matrices
//
//
// Command line:
//   -n  N        matrix size, NxN
//   -nz K        total number of nonzeros
//   -m  mode     regular | irregular
//   -s  seed     optional seed, default 111
//   --dump-vector FILE
//                 optional dump of the final normalized vector
//   -t T         number of threads to use
//   --chunk-size   number of rows to compute contained in a task
//
// Minimal build:
//   g++ -O3 -std=c++20 -I . -Wall openmp_SpMV_worksharing.cpp -o omp_worksharing -fopenmp
//
// Examples:
//   ./omp_worksharing -n 500000 -nz 20000000 -m regular -t 2 --chunk-size 1024
//   ./omp_worksharing -n 500000 -nz 20000000 -m irregular -t 4 --chunk-size 256
//   ./omp_worksharing -n 5000 -nz 20000 -m irregular --dump-vector omp_vec.dump -t 4 --chunk-size 256
//
// Notes:
//   - Matrix generation is not included in computation time.
//   - The computation uses a fixed number of iterations.
//   - The main workload is the irregular case.
//


// always use the same approach when working on a parallel loop:
// all the threads enter the same omp for construct and the iterations are divided among them
// for the SpMV, dynamic scheduling gives a new chunk to a thread as soon as it finishes the previous one
// this is useful because, in the irregular matrix, different rows can contain a very different number of nonzeros
// at the end of every omp for there is the implicit synchronization barrier needed by the next phase



#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "matrix_generation.hpp"
#include "utils.hpp"


// number of iterations
static constexpr std::uint32_t NUM_ITERS = 500;
// number of iterations between two matrix-evolution steps
static constexpr std::uint32_t EPOCH_LEN = 25;


// Vector operations

// this function is called by all the threads of the omp parallel section
static void dot(const std::vector<double>& a, const std::vector<double>& b, std::uint64_t chunk_size, std::vector<padded_double>& partial_norms, double& total_sum) {
    const std::uint64_t n = a.size();
    const std::uint64_t tid = omp_get_thread_num();
    double local_sum = 0.0;

    // each thread computes its local partial sum
    #pragma omp for nowait schedule(static, chunk_size)
    for (std::uint64_t i = 0; i < n; i++) {
        local_sum += a[i] * b[i];
    }

    // one padded value for every thread avoids false sharing between local accumulators
    partial_norms[tid].value = local_sum;

    // wait for all the threads completion before doing the reduction
    #pragma omp barrier

    // here only one thread combines the partial sums
    #pragma omp single
    {
        total_sum = 0.0;
        for (const padded_double& partial_norm : partial_norms) {
            total_sum += partial_norm.value;
        }
    }
}

static void l2_norm(const std::vector<double>& x, std::uint64_t chunk_size, std::vector<padded_double>& partial_norms, double& norm) {
    dot(x, x, chunk_size, partial_norms, norm);

    // after the barrier of dot, one thread computes the square root
    #pragma omp single
    {
        norm = std::sqrt(norm);
    }
}

static void normalize(std::vector<double>& x, std::uint64_t chunk_size, std::vector<padded_double>& partial_norms, double& norm) {
    l2_norm(x, chunk_size, partial_norms, norm);
    const double inv_norm = 1.0 / norm;
    const std::uint64_t n = x.size();

    // apply division to normalize
    // every iteration has the same small amount of work, so static scheduling is enough here
    #pragma omp for schedule(static, chunk_size)
    for (std::uint64_t i = 0; i < n; i++) {
        x[i] *= inv_norm;
    }
}


// computes the epoch parameter
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// Per-row SpMV kernel
// this function is called by all the threads of the omp parallel section
static void spmv_csr_shifted_rows(const CSRMatrix& A, std::size_t row_shift, std::uint64_t chunk_size, const std::vector<double>& x, std::vector<double>& y) {
    const std::uint64_t n = A.n;

    // rows in the irregular matrix have different lengths.
    // dynamic scheduling balances the work by assigning a new chunk to the next free thread.
    #pragma omp for schedule(dynamic, chunk_size)
    for (std::uint64_t i = 0; i < n; i++) {
        const std::uint64_t src_row = (i + n - row_shift) % n;
        double sum = 0.0;
        for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; ++p) {
            sum += A.values[p] * x[A.col_idx[p]];
        }

        y[i] = sum;
    }
}


struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};


static IterativeResult iterative_spmv_evolving(const CSRMatrix& A, std::uint64_t seed, std::uint64_t num_threads, std::uint64_t chunk_size, std::vector<double>* final_vector = nullptr) {
    const std::size_t n = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);

    // Phase 1: initialize the vector used by the iterative method (SEQUENTIAL)
    std::vector<double> x(n);
    std::vector<double> y(n);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : x) {
        v = rng.next_unit();
    }

    // normalization is done in the omp section

    // Phase 2: parallel section
    // global values to return at the end
    std::size_t row_shift = 0;
    double rayleigh = 0.0;
    double norm = 0.0;
    std::uint64_t checksum = 0;

    // partial values: one padded value for every OpenMP thread
    std::vector<padded_double> partial_norms(num_threads);
    std::vector<padded_uint64> partial_checksums(num_threads);

    #pragma omp parallel num_threads(num_threads) default(none) shared(A, x, y, row_shift, rayleigh, norm, checksum, shift_rows, chunk_size, n, partial_norms, partial_checksums)
    {
        // normalization of the initialized x vector
        normalize(x, chunk_size, partial_norms, norm);

        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
            // only one thread does this work
            #pragma omp single
            {
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    row_shift = (row_shift + shift_rows) % n;
                }
            }

            // all the threads cooperate on the iteration of SpMV
            spmv_csr_shifted_rows(A, row_shift, chunk_size, x, y);

            normalize(y, chunk_size, partial_norms, norm);

            // only one thread swaps the two shared vectors after normalization
            #pragma omp single
            {
                x.swap(y);
            }
        }

        // Phase 3: final diagnostics for correctness checks.
        // The extra SpMV is used to compute the final Rayleigh-like value.
        spmv_csr_shifted_rows(A, row_shift, chunk_size, x, y);
        dot(x, y, chunk_size, partial_norms, rayleigh);

        // as in the sequential version, every position contributes to the final checksum.
        // every thread computes a local checksum, then a single thread combines all the partial values.
        const std::uint64_t tid = omp_get_thread_num();
        std::uint64_t local_acc = 0;

        #pragma omp for nowait schedule(static, chunk_size)
        for (std::uint64_t i = 0; i < n; i++) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &x[i], sizeof(double));
            local_acc ^= SplitMix64::mix(bits ^ SplitMix64::mix(i));
        }

        partial_checksums[tid].value = local_acc;

        // the omp for uses nowait, so wait until every thread has stored its local checksum
        #pragma omp barrier

        #pragma omp single
        {
            checksum = 0;
            for (const padded_uint64& partial_checksum : partial_checksums) {
                checksum ^= partial_checksum.value;
            }
        }
    }

    // Keep the final vector only if we have to dump it.
    if (final_vector != nullptr) {
        *final_vector = std::move(x);
    }

    return IterativeResult{
        .rayleigh = rayleigh,
        .checksum = checksum,
        .final_row_shift = row_shift
    };
}

int main(int argc, char** argv) {
    // Phase 0: read problem size, sparsity mode, seed, and optional dump path.
    std::uint64_t n64  = 0;
    std::uint64_t nz   = 0;
    std::uint64_t seed = 111;
    std::uint64_t num_threads = 1;
    std::uint64_t chunk_size = 256;
    std::string mode;
    std::string dump_vector_path;

    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode) ||
        !read_arg_u64(argc, argv, "--chunk-size", chunk_size) ||
        !read_arg_u64(argc, argv, "-t", num_threads)) {
        usage(argv[0]);
        return 1;
    }
    if (num_threads <= 0) {
        std::cerr << "[ERROR] number of threads t must be > 0" << std::endl;
        return 1;
    }
    if (chunk_size <= 0 || chunk_size > n64) {
        std::cerr << "[ERROR] chunk size C must be: 0 < C <= n" << std::endl;
        return 1;
    }

    // Optional arguments
    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);
    std::cout << "SPARSE_ITERATION_OPENMP\n";

    try {
        // Phase 1: input construction.
        const auto tg0 = std::chrono::steady_clock::now();
        const GeneratedMatrix G = generate_matrix(n, nz, seed, mode);
        const auto tg1 = std::chrono::steady_clock::now();

        const double generation_sec = std::chrono::duration<double>(tg1 - tg0).count();

        print_matrix_stats(G);
        std::cout << "generation_time_sec=" << generation_sec << "\n\n";

        std::vector<double>  final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        // Phase 2: timed iterative computation.
        const auto tc0 = std::chrono::steady_clock::now();
        const IterativeResult result = iterative_spmv_evolving(G.A, seed, num_threads, chunk_size, final_vector_out);
        const auto tc1 = std::chrono::steady_clock::now();

        const double computation_sec = std::chrono::duration<double>(tc1 - tc0).count();

        std::cout << std::setprecision(15);
        std::cout << "rayleigh=" << result.rayleigh << "\n";
        std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << "\n";

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Time (sec) = " << computation_sec << "\n";

        // Phase 3: optional correctness support. Vector dumping is outside the timed region
        if (!dump_vector_path.empty()) {
            dump_vector(dump_vector_path, final_vector);
            std::cout << "vector_dump=" << dump_vector_path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
