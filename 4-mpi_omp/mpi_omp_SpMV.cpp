// Hybrid MPI + OpenMP implementation of the Iterative Sparse Matrix-Vector Computation
//
// Command line:
//   mpirun -np P ./mpi_omp -n N -nz K -m mode -t T --chunk-size C
//
// Minimal build:
//   mpic++ -O3 -std=c++20 -I . -Wall mpi_omp_SpMV.cpp -o mpi_omp -fopenmp
//

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <mpi.h>
#include <omp.h>

#include "matrix_generation.hpp"
#include "utils.hpp"

static constexpr std::uint32_t NUM_ITERS = 500;
static constexpr std::uint32_t EPOCH_LEN = 25;

// Helper logico per l'evoluzione
static std::size_t compute_shift_rows(std::size_t n) {
    std::size_t s = n / 16 + 17;
    if ((s % 2) == 0) ++s;
    s %= n;
    if (s == 0) s = 1;
    return s;
}

struct IterativeResult {
    double rayleigh             = 0.0;
    std::uint64_t checksum      = 0;
    std::size_t final_row_shift = 0;
};

static IterativeResult iterative_spmv_evolving(int rank, int num_ranks, const CSRMatrix& global_A, std::uint64_t n, std::uint64_t seed,  std::uint64_t num_threads, std::uint64_t chunk_size, std::vector<double>* final_vector = nullptr) {
    const std::size_t shift_rows = compute_shift_rows(n);
    
    // initialize and distribute data
    
    // divide data across the ranks
    std::vector<std::uint64_t> row_counts(num_ranks), row_displacements(num_ranks);
    std::uint64_t reminder = n % num_ranks;
    std::uint64_t curr_displacement = 0;
    for (int i = 0; i < num_ranks; i++) {
        row_counts[i] = (n / num_ranks) + (i < reminder ? 1 : 0); // number of PHYSICAL rows to work on
        row_displacements[i] = curr_displacement;
        curr_displacement += row_counts[i];
    }
    
    std::uint64_t local_n = row_counts[rank]; // each rank has its own portion of data to locally work on
    
    // count the number of non zero numbers for each rank based on the previous rows assigned to each rank
    std::vector<std::uint64_t> nnz_counts(num_ranks), nnz_displacements(num_ranks);
    if (rank == 0) {
        for(int i = 0; i < num_ranks; i++) {
            std::uint64_t start_row = row_displacements[i];
            std::uint64_t end_row = start_row + row_counts[i];
            std::uint64_t start_nnz = global_A.row_ptr[start_row];
            std::uint64_t end_nnz = global_A.row_ptr[end_row];
            nnz_counts[i] = end_nnz - start_nnz;
            nnz_displacements[i] = start_nnz;
        }
    }

    std::uint64_t local_nnz;
    MPI_Scatter(nnz_counts.data(), 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD); // pass the non zero elements to work on to each rank

    // local csr structures needed for each rank
    std::vector<double> local_values(local_nnz);
    std::vector<std::uint64_t> local_col_idx(local_nnz);
    std::vector<std::uint64_t> local_row_ptr(local_n + 1);

    // distribute original A matrix's data across ranks
    MPI_Scatterv(rank == 0 ? global_A.values.data() : nullptr, nnz_counts.data(), nnz_displacements.data(), MPI_DOUBLE, local_values.data(), local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? global_A.col_idx.data() : nullptr, nnz_counts.data(), nnz_displacements.data(), MPI_UINT32_T, local_col_idx.data(), local_nnz, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    // send overlapping row_ptr data
    std::vector<std::uint64_t> ptr_counts(num_ranks), ptr_displs(num_ranks);
    if (rank == 0) {
        for(int i = 0; i < num_ranks; i++) {
            ptr_counts[i] = row_counts[i] + 1;
            ptr_displs[i] = row_displacements[i];
        }
    }
    MPI_Scatterv(rank == 0 ? global_A.row_ptr.data() : nullptr, ptr_counts.data(), ptr_displs.data(), MPI_UINT64_T, local_row_ptr.data(), local_n + 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    std::uint64_t offset = local_row_ptr[0];
    for (std::uint64_t i = 0; i <= local_n; i++) {
        local_row_ptr[i] -= offset; // local row_ptr should start from 0
    }

    

    // local SpMV for each rank


    std::vector<double> x(n);
    std::vector<double> y(n);
    
    std::vector<double> local_out(local_n);
    std::vector<double> global_out(n);

    SplitMix64 rng(seed ^ 0x123456789abcdef0ULL);
    for (double& v : x) {
        v = rng.next_unit(); // each rank creates its own x vector locally (since rng is created locally with the same seed)
    }

    std::size_t row_shift = 0;
    double rayleigh = 0.0;
    std::uint64_t checksum = 0;

    
    #pragma omp parallel num_threads(num_threads) shared(x, y, local_out, global_out, row_shift, rayleigh, checksum)
    {
        double norm2 = 0.0;
        #pragma omp for reduction(+:norm2)
        for (std::size_t i = 0; i < n; ++i) {
            norm2 += x[i] * x[i];
        }
        double initial_norm = std::sqrt(norm2);
        
        #pragma omp for
        for (std::size_t i = 0; i < n; ++i) {
            x[i] /= initial_norm;
        }
        

        for (std::uint32_t iter = 0; iter < NUM_ITERS; ++iter) {
            #pragma omp single
            {
                if (iter > 0 && (iter % EPOCH_LEN) == 0) {
                    row_shift = (row_shift + shift_rows) % n;
                }
            }

            #pragma omp single
            {
                #pragma omp taskloop grainsize(chunk_size) default(none) shared(local_row_ptr, local_col_idx, local_values, x, local_out) firstprivate(local_n)
                for (std::uint64_t i = 0; i < local_n; ++i) {
                    double sum = 0.0;
                    for (std::uint64_t p = local_row_ptr[i]; p < local_row_ptr[i + 1]; ++p) {
                        sum += local_values[p] * x[local_col_idx[p]];
                    }
                    local_out[i] = sum;
                }
            }
            
            // synchronization before communication
            #pragma omp barrier

            #pragma omp single
            {
                // gather local results into a unique global result
                MPI_Allgatherv(local_out.data(), local_n, MPI_DOUBLE, global_out.data(), row_counts.data(), row_displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            }

            double iter_normsq = 0.0;
            #pragma omp for reduction(+:iter_normsq)
            for (std::size_t p = 0; p < n; ++p) {
                double val = global_out[p];
                y[(p + row_shift) % n] = val; // !!!!!!!!!!!!!!!!! check
                iter_normsq += val * val;
            }
            
            double iter_norm = std::sqrt(iter_normsq);
            double inv_norm = 1/iter_norm;
            
            #pragma omp for
            for (std::size_t i = 0; i < n; ++i) {
                y[i] *= inv_norm;
            }

            #pragma omp single
            {
                x.swap(y);
            }
        }

        
        // rayleigh and checksum computation
        #pragma omp single
        {
            #pragma omp taskloop grainsize(chunk_size) default(none) shared(local_row_ptr, local_col_idx, local_values, x, local_out) firstprivate(local_n)
            for (std::uint64_t i = 0; i < local_n; ++i) {
                double sum = 0.0;
                for (std::uint64_t p = local_row_ptr[i]; p < local_row_ptr[i + 1]; ++p) {
                    sum += local_values[p] * x[local_col_idx[p]];
                }
                local_out[i] = sum;
            }
        }
        #pragma omp barrier

        #pragma omp single
        {
            MPI_Allgatherv(local_out.data(), local_n, MPI_DOUBLE, global_out.data(), row_counts.data(), row_displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        }

        #pragma omp for reduction(+:rayleigh)
        for (std::size_t p = 0; p < n; ++p) {
            double val = global_out[p];
            y[(p + row_shift) % n] = val;
            rayleigh += x[(p + row_shift) % n] * val;
        }

        #pragma omp for reduction(^:checksum)
        for (std::size_t i = 0; i < n; ++i) {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &x[i], sizeof(double));
            checksum ^= SplitMix64::mix(bits ^ SplitMix64::mix(i));
        }

    }

    if (final_vector != nullptr && rank == 0) {
        *final_vector = std::move(x);
    }

    return IterativeResult{
        .rayleigh = rayleigh,
        .checksum = checksum,
        .final_row_shift = row_shift
    };
}

int main(int argc, char** argv) {
    int provided;
    // why mpi thread funneled????
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

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
        
        // only rank 0 creates the matrix A
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
        const auto tc0 = std::chrono::steady_clock::now();
        
        const IterativeResult result = iterative_spmv_evolving(rank, size, G.A, n, seed, num_threads, chunk_size, final_vector_out);
        
        MPI_Barrier(MPI_COMM_WORLD);
        const auto tc1 = std::chrono::steady_clock::now();

        if (rank == 0) {
            const double computation_sec = std::chrono::duration<double>(tc1 - tc0).count();

            std::cout << std::setprecision(15);
            std::cout << "rayleigh=" << result.rayleigh << "\n";
            std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << "\n";

            std::cout << std::fixed << std::setprecision(6);
            std::cout << "Time (sec) = " << computation_sec << "\n";

            if (!dump_vector_path.empty()) {
                dump_vector(dump_vector_path, final_vector);
                std::cout << "vector_dump=" << dump_vector_path << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] rank " << rank << ": " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}