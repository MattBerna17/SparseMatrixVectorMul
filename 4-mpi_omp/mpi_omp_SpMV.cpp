// Hybrid MPI + OpenMP implementation of the Iterative Sparse Matrix-Vector Computation
//
// Command line:
//   mpirun -n P ./mpi_omp -n N -nz K -m mode -t T --chunk-size C
//
// Minimal build:
//   mpic++ -O3 -std=c++20 -I . -Wall mpi_omp_SpMV.cpp -o mpi_omp -fopenmp
//
// Examples:
//   salloc -N 2
//   mpirun -n 2 ./mpi_omp -n 500000 -nz 20000000 -m regular -t 4 --chunk-size 1024
//   mpirun -n 4 ./mpi_omp -n 500000 -nz 20000000 -m irregular -t 2 --chunk-size 256
//   mpirun -n 4 ./mpi_omp -n 5000 -nz 20000 -m irregular -t 2 --chunk-size 256 --dump-vector omp_vec.dump
//
// Notes:
//   - Matrix generation is not included in computation time.
//   - The computation uses a fixed number of iterations.
//   - The main workload is the irregular case.
//

// rank 0 initially defines the assignment of rows to ranks that minimizes the distance to the ideal assignment (assigning desired_nnz_per_rank to each rank) counting the number of nnz for each rank
// then, rank 0 shares the csr data to the ranks, and then each rank computes its own local spmv
// the results are then gathered together, and the y result vector is created
// then the y vector is rotated according to the current value of row_shift, and gets normalized
// as the last step, swap x and y, then go to the next iteration
// the time is measured for each phase of the execution and is reduced using the max operator (execution time in parallel is the max of the executions of the single ranks)

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include <omp.h>

#include "matrix_generation.hpp"
#include "utils.hpp"

// number of iterations
static constexpr std::uint32_t NUM_ITERS = 500;
// number of iterations between two matrix-evolution steps
static constexpr std::uint32_t EPOCH_LEN = 25;

// Computes the epoch parameter
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

// Vector operations

static double dot(const std::vector<double>& a, const std::vector<double>& b, std::uint64_t num_chunks, std::uint64_t chunk_size) {
    std::uint64_t n = a.size();
    std::vector<padded_double> partial_norms(num_chunks);
    
    for (std::uint64_t i = 0; i < num_chunks; i++) {
        std::uint64_t start = chunk_size * i;
        std::uint64_t end = std::min(start + chunk_size, n);

        #pragma omp task default(none) shared(a, b, partial_norms) firstprivate(i, start, end)
        {
            double local_sum = 0.0;
            for (std::uint64_t j = start; j < end; j++) {
                local_sum += a[j] * b[j];
            }
            partial_norms[i].value = local_sum;
        }
    }

    // barrier for the threads computing the tasks
    #pragma omp taskwait

    // here only the single thread of the "omp single" block works
    double total_sum = 0.0;
    for (std::uint64_t i = 0; i < num_chunks; i++) {
        total_sum += partial_norms[i].value;
    }

    return total_sum;
}

static double l2_norm(const std::vector<double>& x, std::uint64_t num_chunks, std::uint64_t chunk_size) {
    return std::sqrt(dot(x, x, num_chunks, chunk_size));
}

static void normalize(std::vector<double>& x, std::uint64_t num_chunks, std::uint64_t chunk_size) {
    double norm = l2_norm(x, num_chunks, chunk_size);
    double inv_norm = 1.0 / norm;
    std::uint64_t n = x.size();

    // apply division to normalize
    for (std::uint64_t i = 0; i < num_chunks; i++) {
        std::uint64_t start = chunk_size * i;
        std::uint64_t end = std::min(start + chunk_size, n);

        #pragma omp task default(none) shared(x) firstprivate(inv_norm, start, end)
        {
            for (std::uint64_t j = start; j < end; j++) {
                x[j] *= inv_norm;
            }
        }
    }

    #pragma omp taskwait
}

// Per-row local SpMV kernel
static void local_spmv_csr(int local_n, const std::vector<std::uint64_t>& local_row_ptr, const std::vector<std::uint32_t>& local_col_idx, const std::vector<double>& local_values, const std::vector<double>& x, std::vector<double>& local_out, int num_local_chunks, int chunk_size) {
    for (int chunk = 0; chunk < num_local_chunks; chunk++) {
        int start = chunk_size * chunk;
        int end = std::min(start + chunk_size, local_n);

        #pragma omp task default(none) shared(local_row_ptr, local_col_idx, local_values, x, local_out) firstprivate(start, end)
        {
            for (int i = start; i < end; i++) {
                double sum = 0.0;
                for (std::uint64_t p = local_row_ptr[i]; p < local_row_ptr[i + 1]; ++p) {
                    sum += local_values[p] * x[local_col_idx[p]];
                }
                local_out[i] = sum;
            }
        }
    }
    #pragma omp taskwait 
}

// rotate and transfer global results logic
static void rotate_vector(const std::vector<double>& global_out, std::vector<double>& y, std::size_t row_shift, std::uint64_t num_chunks, std::uint64_t chunk_size) {
    std::uint64_t n = global_out.size();
    for (std::uint64_t chunk = 0; chunk < num_chunks; chunk++) {
        std::uint64_t start = chunk_size * chunk;
        std::uint64_t end = std::min(start + chunk_size, n);

        #pragma omp task default(none) shared(global_out, y) firstprivate(start, end, row_shift, n)
        {
            for (std::uint64_t i = start; i < end; i++) {
                y[(i + row_shift) % n] = global_out[i]; // P(Ax) = P(A)x
            }
        }
    }
    #pragma omp taskwait
}

struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
    double global_epoch = 0.0;
    double global_spmv = 0.0;
    double global_comm = 0.0;
    double global_rotate = 0.0;
    double global_normalize = 0.0;
};

static IterativeResult distributed_iterative_spmv(int rank, int num_ranks, const CSRMatrix& global_A, std::uint64_t nz, std::uint64_t n, std::uint64_t seed, std::uint64_t num_threads, std::uint64_t chunk_size, std::vector<double>* final_vector = nullptr) {
    const std::size_t shift_rows = compute_shift_rows(n);
    
    // initialize and distribute data
    
    // divide data across the ranks (just define the number of rows to send to each rank)
    std::vector<int> row_counts(num_ranks), row_displacements(num_ranks);
    
    if (rank == 0) {
        // starting from the row 0, assign each row to a rank
        // the optimal partitioning is assigning desired_nnz_per_rank nnzs to each rank (not always possibile if A is generated using an irregular pattern)
        std::uint64_t curr_row = 0;
        std::uint64_t curr_nnz = 0;
        std::uint64_t desired_nnz_per_rank = nz / num_ranks;
        // std::vector<uint64_t> row_nnz(n);
        std::uint64_t curr_rank = 0;
        for (size_t i = 0; i < n; i++) {
            auto row_nnz = global_A.row_ptr[i+1] - global_A.row_ptr[i];
            if (curr_nnz + row_nnz > desired_nnz_per_rank && curr_rank < static_cast<std::uint64_t>(num_ranks - 1)) {
                // custom function to return absolute difference in case of uint64
                auto abs_diff = [](std::uint64_t a, std::uint64_t b) -> std::uint64_t {
                    return a > b ? a - b : b - a;
                };
                if (abs_diff(desired_nnz_per_rank, curr_nnz) < abs_diff(desired_nnz_per_rank, curr_nnz + row_nnz)) {
                    row_counts[curr_rank] = i - curr_row;
                    row_displacements[curr_rank] = curr_row;
                    curr_row = i;
                    curr_rank += 1;
                    curr_nnz = 0;
                }
            }
            curr_nnz += row_nnz;
        }
        // for last rank (get the remaining rows)
        row_counts[curr_rank] = n - curr_row;
        row_displacements[curr_rank] = curr_row;
    }

    // no scatter because, later on, i need the complete row_counts and row_displacements arrays (easier to use broadcast in this case)
    MPI_Bcast(row_counts.data(), num_ranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_displacements.data(), num_ranks, MPI_INT, 0, MPI_COMM_WORLD);
    
    int local_n = row_counts[rank];

    // count the number of non zero numbers for each rank based on the previous rows assigned to each rank
    std::vector<int> nnz_counts(num_ranks), nnz_displacements(num_ranks);
    if (rank == 0) {
        for (int i = 0; i < num_ranks; i++) {
            std::uint64_t start_nnz = global_A.row_ptr[row_displacements[i]]; // initial row pointer
            std::uint64_t end_nnz = global_A.row_ptr[row_displacements[i] + row_counts[i]]; // final row pointer
            nnz_counts[i] = static_cast<int>(end_nnz - start_nnz);
            nnz_displacements[i] = static_cast<int>(start_nnz);
        }
    }

    int local_nnz;
    // pass the non zero elements to work on to each rank
    MPI_Scatter(nnz_counts.data(), 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD); 

    // local csr structures needed for each rank
    std::vector<double> local_values(local_nnz);
    std::vector<std::uint32_t> local_col_idx(local_nnz);
    std::vector<std::uint64_t> local_row_ptr(local_n + 1);

    // distribute original A matrix's data across ranks
    MPI_Scatterv(rank == 0 ? global_A.values.data() : nullptr, nnz_counts.data(), nnz_displacements.data(), MPI_DOUBLE, local_values.data(), local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? global_A.col_idx.data() : nullptr, nnz_counts.data(), nnz_displacements.data(), MPI_UINT32_T, local_col_idx.data(), local_nnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // send overlapping row_ptr data
    std::vector<int> ptr_counts(num_ranks), ptr_displacements(num_ranks);
    if (rank == 0) {
        for (int i = 0; i < num_ranks; i++) {
            ptr_counts[i] = row_counts[i] + 1;
            ptr_displacements[i] = row_displacements[i];
        }
    }
    MPI_Scatterv(rank == 0 ? global_A.row_ptr.data() : nullptr, ptr_counts.data(), ptr_displacements.data(), MPI_UINT64_T, local_row_ptr.data(), local_n + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    std::uint64_t offset = local_row_ptr[0];
    for (int i = 0; i <= local_n; i++) {
        local_row_ptr[i] -= offset; // local
    }


    // times to measure
    double spmv_time = 0.0;
    double comm_time = 0.0;
    double rotate_time = 0.0;
    double normalize_time = 0.0;
    double epoch_time = 0.0;


    // Phase 1: initialize the vector used by the iterative method (SEQUENTIAL)
    std::vector<double> x(n);
    std::vector<double> y(n);
    std::vector<double> local_out(local_n);
    std::vector<double> global_out(n);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : x) {
        v = rng.next_unit();
    }

    // normalization is done in the omp section

    // Phase 2: parallel section
    // global values to return at the end
    std::size_t row_shift = 0;
    double rayleigh = 0.0;
    std::uint64_t checksum = 0;

    std::uint64_t num_chunks = (n + chunk_size - 1) / chunk_size;
    int num_local_chunks = (local_n + chunk_size - 1) / chunk_size;
    
    // partial values: one value for each chunk the data is divided into
    std::vector<padded_uint64> partial_checksums(num_chunks);

    #pragma omp parallel num_threads(num_threads) shared(x, y, local_out, global_out, local_values, local_col_idx, local_row_ptr, partial_checksums, row_shift, rayleigh, checksum, num_chunks, num_local_chunks, local_n, row_counts, row_displacements)
    {
        // only the master thread enters here: MPI_THREAD_FUNNELED requires the master thread to execute the MPI calls only (NOT single)
        #pragma omp master
        {
            // normalization of the initialized x vector
            normalize(x, num_chunks, chunk_size);
            // at the end of normalize, there is a barrier, so here we only have 1 thread going on

            for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
                
                auto t0 = std::chrono::steady_clock::now();
                // only single thread does this work
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    row_shift = (row_shift + shift_rows) % n;
                }
                auto t1 = std::chrono::steady_clock::now();
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    epoch_time += std::chrono::duration<double>(t1-t0).count();
                }


                t0 = std::chrono::steady_clock::now();
                // generate tasks for the iteration of SpMV
                local_spmv_csr(local_n, local_row_ptr, local_col_idx, local_values, x, local_out, num_local_chunks, chunk_size);
                t1 = std::chrono::steady_clock::now();
                spmv_time += std::chrono::duration<double>(t1-t0).count();

                double t_i, t_f;
                t_i = MPI_Wtime();
                // gather results into a unique global result
                MPI_Allgatherv(local_out.data(), local_n, MPI_DOUBLE, global_out.data(), row_counts.data(), row_displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
                t_f = MPI_Wtime();
                comm_time += t_f - t_i;
                
                t0 = std::chrono::steady_clock::now();
                // distribute logic with shift rotation via tasks
                rotate_vector(global_out, y, row_shift, num_chunks, chunk_size);
                t1 = std::chrono::steady_clock::now();
                rotate_time += std::chrono::duration<double>(t1-t0).count();
                
                t0 = std::chrono::steady_clock::now();
                normalize(y, num_chunks, chunk_size);
                // barrier -> only one thread from here
                t1 = std::chrono::steady_clock::now();
                normalize_time += std::chrono::duration<double>(t1-t0).count();
                
                x.swap(y);
            }

            // Phase 3: final diagnostics for correctness checks.
            // The extra SpMV is used to compute the final Rayleigh-like value.
            local_spmv_csr(local_n, local_row_ptr, local_col_idx, local_values, x, local_out, num_local_chunks, chunk_size);
            
            MPI_Allgatherv(local_out.data(), local_n, MPI_DOUBLE, global_out.data(), row_counts.data(), row_displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            
            rotate_vector(global_out, y, row_shift, num_chunks, chunk_size);

            rayleigh = dot(x, y, num_chunks, chunk_size);
            
            for (std::uint64_t i = 0; i < num_chunks; i++) {
                std::uint64_t start = chunk_size * i;
                std::uint64_t end = std::min(start + chunk_size, static_cast<std::uint64_t>(n));

                #pragma omp task default(none) shared(x, partial_checksums) firstprivate(i, start, end)
                {
                    std::uint64_t local_acc = 0;
                    for (std::uint64_t j = start; j < end; j++) {
                        std::uint64_t bits = 0;
                        std::memcpy(&bits, &x[j], sizeof(double));
                        local_acc ^= SplitMix64::mix(bits ^ SplitMix64::mix(j));
                    }
                    partial_checksums[i].value = local_acc;
                }
            }
            #pragma omp taskwait

            for (std::uint64_t i = 0; i < num_chunks; i++) {
                checksum ^= partial_checksums[i].value;
            }
        }
    }

    // get global time for each phase (take max since they work in parallel)
    double global_epoch, global_spmv, global_comm, global_rotate, global_normalize;
    MPI_Reduce(&epoch_time, &global_epoch, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&spmv_time, &global_spmv, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &global_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rotate_time, &global_rotate, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&normalize_time, &global_normalize, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Keep the final vector only if we have to dump it.
    if (final_vector != nullptr && rank == 0) {
        *final_vector = std::move(x);
    }

    return IterativeResult{
        .rayleigh = rayleigh,
        .checksum = checksum,
        .final_row_shift = row_shift,
        .global_epoch = global_epoch,
        .global_spmv = global_spmv,
        .global_comm = global_comm,
        .global_rotate = global_rotate,
        .global_normalize = global_normalize
    };
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Error on MPI_Init_thread" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        if (rank == 0) usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    if (num_threads <= 0) {
        if (rank == 0) std::cerr << "[ERROR] number of threads t must be > 0" << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (chunk_size <= 0 || chunk_size > n64) {
        if (rank == 0) std::cerr << "[ERROR] chunk size C must be: 0 < C <= n" << std::endl;
        MPI_Finalize();
        return 1;
    }

    (void)read_arg_u64(argc, argv, "-s", seed);
    (void)read_arg_str(argc, argv, "--dump-vector", dump_vector_path);

    const std::size_t n = static_cast<std::size_t>(n64);
    
    if (rank == 0) {
        std::cout << "SPARSE_ITERATION_MPI_OPENMP_HYBRID\n";
        std::cout << "MPI Processes: " << size << ", Threads/Proc: " << num_threads << ", Chunk Size: " << chunk_size << "\n";
    }

    try {
        GeneratedMatrix G;
        double generation_sec = 0;
        
        // generate starting matrix
        if (rank == 0) {
            const auto tg0 = std::chrono::steady_clock::now();
            G = generate_matrix(n, nz, seed, mode);
            const auto tg1 = std::chrono::steady_clock::now();
            generation_sec = std::chrono::duration<double>(tg1 - tg0).count();
            
            print_matrix_stats(G);
            std::cout << "generation_time_sec=" << generation_sec << "\n\n";
        }

        std::vector<double> final_vector;
        std::vector<double>* final_vector_out = dump_vector_path.empty() ? nullptr : &final_vector;

        // barrier to make sure each rank is at the same point before counting time
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Phase 2: timed iterative computation.
        const auto tc0 = std::chrono::steady_clock::now(); // i still measure the distribution of the matrix etc because it's always necessary when executing with MPI+OMP
        
        const IterativeResult result = distributed_iterative_spmv(rank, size, G.A, nz, n, seed, num_threads, chunk_size, final_vector_out);
        
        MPI_Barrier(MPI_COMM_WORLD);
        const auto tc1 = std::chrono::steady_clock::now();

        if (rank == 0) {
            const double computation_sec = std::chrono::duration<double>(tc1 - tc0).count();

            std::cout << std::setprecision(15);
            std::cout << "rayleigh=" << result.rayleigh << "\n";
            std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << "\n";

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Time (sec)" << std::endl << "computation=" << computation_sec << "\n";
            std::cout << "global_spmv=" << result.global_spmv << std::endl;
            std::cout << "global_epoch=" << result.global_epoch << std::endl;
            std::cout << "global_comm=" << result.global_comm << std::endl;
            std::cout << "global_rotate=" << result.global_rotate << std::endl;
            std::cout << "global_normalize=" << result.global_normalize << std::endl;

            // Phase 3: optional correctness support. Vector dumping is outside the timed region
            if (!dump_vector_path.empty()) {
                dump_vector(dump_vector_path, final_vector);
                std::cout << "vector_dump=" << dump_vector_path << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error on rank " << rank << ": " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}