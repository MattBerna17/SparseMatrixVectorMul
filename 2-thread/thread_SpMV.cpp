// Thread version of the Sparse Matrix-Vector multiplication

// tools used:
//      threadpool: because of the fixed number of elements (A is nxn, x is n, y is n) with a fixed number of threads (<= n, TO EXPLORE, but still power of 2)
//      packaged_task: to create the tasks to solve
//      futures and promises?

// questions:
//      what synchronization primitive should i use? semaphores? cvs? the impact?
//      synchronization barriers between two particular steps

// to analyze with experiments:
//      best number of threads to use
//      impact of nnz and mode (for workload per thread: try static with cyclical and block, but also dynamic, see which one performs better -> expected static with block because of locality and low synchronization overhead)
//      optimizations: consider the cache's block dimension and try cold vs warm approach
//      strong & weak scaling

//
// Command line:
//   -n  N          matrix size, NxN
//   -nz K          total number of nonzeros
//   -m  mode       regular | irregular
//   -s  seed       optional seed, default 111
//   --dump-vector  FILE
//                  optional dump of the final normalized vector
//   -t T           number of threads to use
//   --scheduling   static | dynamic
//   --chunk-size   number of elements to compute contained in a chunk (task assigned to a thread). ONLY WORKS WITH DYNAMIC SCHEDULING, with static chunk-size = N / T with remainder for the first N % T threads
//
// Minimal build:
//   g++ -O3 -std=c++20 -I . -Wall thread_SpMV.cpp -o thread
//
// Examples:
//   ./thread -n 500000 -nz 20000000 -m regular -t 2 --scheduling static
//   ./thread -n 500000 -nz 20000000 -m irregular -t 4 --scheduling dynamic --chunk-size 1024
//   ./thread -n 5000 -nz 20000 -m irregular -t 2 --scheduling dynamic --chunk-size 1024 --dump-vector thread_vec.dump
//
// Notes:
//   - Matrix generation is not included in computation time. --> not optimized, has its own timer, different from the iterative algorithm.
//   - The computation uses a fixed number of iterations.
//   - The main workload is the irregular case.
//

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

static double dot(const std::vector<double>& a, const std::vector<double>& b, ThreadPool &tp, SchedulingParameters &sp) {
    double sum = 0;
    std::vector<padded_double> partial_sums(tp.num_threads);
    std::atomic<std::uint64_t> next_chunk{0};
    
    if (!sp.is_dynamic) {
        // in case of static scheduling
        for (std::uint64_t t = 0; t < tp.num_threads; t++) {
            auto [start, end] = get_static_range(a.size(), tp.num_threads, t);
            
            tp.enqueue([&a, &b, t, &partial_sums, start, end] {
                double local_sum = 0;
                for (std::uint64_t i = start; i < end; i++) {
                    local_sum += (a[i] * b[i]);
                }
                partial_sums[t].value = local_sum; // add the local_sum variable because of cache invalidity
            });
        }
    } else {
        // in case of dynamic scheduling with chunk size defined
        std::uint64_t chunk_size = sp.chunk_size;
        std::uint64_t num_chunks = (a.size() + chunk_size - 1) / chunk_size;
        std::uint64_t n = a.size();
        for (std::uint64_t t = 0; t < tp.num_threads; t++) {
            tp.enqueue([&a, &b, t, &partial_sums, &next_chunk, n, chunk_size, num_chunks] {
                double local_sum = 0;
                while (true) {
                    std::uint64_t chunk = next_chunk.fetch_add(1, std::memory_order_relaxed);
                    if (chunk >= num_chunks) {
                        break;
                    }
                    std::uint64_t start = chunk * chunk_size;
                    std::uint64_t end = std::min(start + chunk_size, n);
                    for (std::uint64_t i = start; i < end; i++) {
                        local_sum += (a[i] * b[i]);
                    }
                }
                partial_sums[t].value = local_sum;
            });
        }

    }

    tp.barrier();

    for (std::uint64_t t = 0; t < tp.num_threads; t++) {
        sum += partial_sums[t].value;
    }

    return sum;
}

static double l2_norm(const std::vector<double>& x, ThreadPool &tp, SchedulingParameters &sp) {
    return std::sqrt(dot(x, x, tp, sp));
}

static void normalize(std::vector<double>& x, ThreadPool &tp, SchedulingParameters &sp) {
    const double nrm = l2_norm(x, tp, sp);
    const double inv = 1.0 / nrm;

    for (std::uint64_t t = 0; t < tp.num_threads; t++) {
        auto [start, end] = get_static_range(x.size(), tp.num_threads, t);

        tp.enqueue([&x, t, start, end, inv] {
            for (std::uint64_t i = start; i < end; i++) {
                x[i] *= inv;
            }
        });
    }

    tp.barrier();
}


// Computes the epoch parameter
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// Per-row SpMV kernel.
//
// row_shift = s means that the matrix rows have been circularly shifted by s positions
//
// Logical row i uses source row (i - s mod n) from the original CSR matrix.
static void spmv_csr_shifted_rows(const CSRMatrix& A, std::size_t row_shift, const std::vector<double>& x, std::vector<double>& y, ThreadPool &tp, SchedulingParameters &sp) {
    const std::uint64_t n = A.n;

    // initialize y to all 0s
    std::uint64_t T = tp.num_threads;

    auto do_row = [&](std::uint64_t i) {
        const std::size_t src_row = (i + n - row_shift) % n;
        
        double sum = 0.0;
        for (std::uint64_t p = A.row_ptr[src_row]; p < A.row_ptr[src_row + 1]; ++p) {
            sum += A.values[p] * x[A.col_idx[p]];
        }

        y[i] = sum;
    };
    
    std::atomic<std::size_t> next_chunk{0};

    if (!sp.is_dynamic) {
        for (std::uint64_t t = 0; t < T; ++t) {
            auto [start, end] = get_static_range(n, T, t);
            tp.enqueue([do_row, start, end] {
                for (std::uint64_t i = start; i < end; i++) {
                    do_row(i);
                }
            });
        }
    } else {
        const std::uint64_t chunk_size = sp.chunk_size;
        const std::uint64_t num_chunks = (n + chunk_size - 1) / chunk_size;

        for (std::uint64_t t = 0; t < T; t++) {
            tp.enqueue([do_row, &next_chunk, n, chunk_size, num_chunks] {
                while (true) {
                    const std::uint64_t chunk = next_chunk.fetch_add(1, std::memory_order_relaxed);
                    if (chunk >= num_chunks) {
                        break;
                    }
                    const std::uint64_t start = chunk * chunk_size;
                    const std::uint64_t end = std::min(start + chunk_size, n);
                    for (std::uint64_t i = start; i < end; i++) {
                        do_row(i);
                    }
                }
            });
        }
    }

    tp.barrier();
}

struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};

static IterativeResult iterative_spmv_evolving(const CSRMatrix& A, std::uint64_t seed, ThreadPool &tp, SchedulingParameters &sp, std::vector<double>* final_vector = nullptr) {
    const std::size_t n = A.n;
    const std::size_t shift_rows = compute_shift_rows(n);

    // PHASE 1: initialize the vector used by the iterative method.
    // Parallel versions must preserve this initialization, or distribute the same initial vector, before entering the timed iterative loop.
    std::vector<double> x(n);
    std::vector<double> y(n);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    // do not parallelize since ordering wouldn't be preserved. in the sequential version, the order would be preserved, but using threads the calls to rng are not ordered, therefore would result in different x vectors between sequential and parallel versions
    for (double& v : x) {
        v = rng.next_unit();
    }
    normalize(x, tp, sp);

    // PHASE 2: iterative computation on the evolving matrix.
    // The sequential reference keeps the CSR matrix fixed and represents matrix
    // evolution through this logical row_shift value.
    std::size_t row_shift = 0;

    for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
        // At each epoch boundary, update the logical row mapping.
        if (iter > 0 && (iter % EPOCH_LEN) == 0) {
            row_shift = (row_shift + shift_rows) % n;
        }

        // One iteration: shifted SpMV followed by vector normalization.
        // The normalization contains a global reduction.
        spmv_csr_shifted_rows(A, row_shift, x, y, tp, sp);
        normalize(y, tp, sp);

        x.swap(y); // O(1), do not need to parallelize
    }

    // PHASE 3: final diagnostics for correctness checks.
    // The extra SpMV is used to compute the final Rayleigh-like value.
    spmv_csr_shifted_rows(A, row_shift, x, y, tp, sp);
    const double rayleigh = dot(x, y, tp, sp);
    const std::uint64_t checksum = checksum_vector(x, tp);

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
    std::string mode;
    std::string dump_vector_path;
    std::uint64_t num_threads = 0;
    std::string scheduling = "static";
    std::uint64_t chunk_size = 0;

    if (!read_arg_u64(argc, argv, "-n", n64) ||
        !read_arg_u64(argc, argv, "-nz", nz) ||
        !read_arg_str(argc, argv, "-m", mode) ||
        !read_arg_u64(argc, argv, "-t", num_threads)) {
        usage(argv[0]);
        return 1;
    }

    // Optional arguments
    SchedulingParameters sp;
    (void)read_arg_str(argc, argv, "--scheduling", scheduling);
    if (scheduling == "dynamic") {
        sp.is_dynamic = true;
        if (!read_arg_u64(argc, argv, "--chunk-size", chunk_size) || chunk_size <= 0) {
            std::cerr << "[ERROR] --scheduling dynamic requires --chunk-size C with C > 0" << std::endl;
            return 1;
        }
        sp.chunk_size = chunk_size;
    } else if (scheduling != "static") {
        std::cerr << "[ERROR] --scheduling must be 'static' or 'dynamic'" << std::endl;
        return 1;
    }

    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    if (n64 < num_threads || num_threads <= 0) {
        std::cerr << "[ERROR] number of threads should be 0 < T <= n" << std::endl;
        return 1;
    }

    const std::size_t n = static_cast<std::size_t>(n64);
    std::cout << "SPARSE_ITERATION_THREADPOOL\n";

    try {
        ThreadPool tp(num_threads);

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
        const IterativeResult result = iterative_spmv_evolving(G.A, seed, tp, sp, final_vector_out);
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
