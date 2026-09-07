# Iterative SpMV 

Four implementations of an iterative sparse matrix-vector computation with
periodic row shifts: sequential reference, C++ threads (custom ThreadPool),
OpenMP (task-based and, optionally, work-sharing), and MPI+OpenMP.

See `report.pdf` for implementation strategies, correctness methodology, and
the full experimental analysis.

## Project structure

```
1-sequential/   sequential reference (iterative_SpMV.cpp)
2-thread/       C++ threads version, custom ThreadPool (thread_SpMV.cpp)
3-openmp/       OpenMP task-based (openmp_SpMV.cpp) and work-sharing
                (openmp_SpMV_worksharing.cpp) versions
4-mpi_omp/      MPI+OpenMP distributed version (mpi_omp_SpMV.cpp)
experiments_scripts/  scripts used to run the experiments and produce the
                       plots discussed in the report
```

## Build
Inside the target node acquired via `salloc -w node0x --time=hh:mm:ss` and then enter it using `ssh node0x`. Next, execute the following commands:

```
make        # builds seq, thread, omp, omp_worksharing, mpi_omp
make clean
```

Requires a C++20 compiler and an MPI implementation providing `mpic++`.
Executables are placed inside each version's directory (e.g. `2-thread/thread`).

## Running

All executables share the same core arguments:

```
-n N        matrix size, N x N
-nz K       total number of nonzeros
-m mode     regular | irregular
-s seed     optional, default 111
```

Thread and OpenMP versions additionally take `-t` (number of workers) and
`--chunk-size` (task granularity). The thread version also takes
`--scheduling static|dynamic`. MPI+OpenMP is launched with `mpirun`:

```
./1-sequential/seq -n 500000 -nz 20000000 -m irregular
./2-thread/thread -n 500000 -nz 20000000 -m irregular -t 8 --scheduling dynamic --chunk-size 1024
./3-openmp/omp -n 500000 -nz 20000000 -m irregular -t 8 --chunk-size 1024
./3-openmp/omp_worksharing -n 500000 -nz 20000000 -m irregular -t 8 --chunk-size 1024
mpirun -n 4 ./4-mpi_omp/mpi_omp -n 500000 -nz 20000000 -m irregular -t 4 --chunk-size 1024
```

Each run prints the Rayleigh value, a checksum, and the measured execution
time. The MPI+OpenMP version additionally reports the per-phase time
breakdown (local computation, communication, reduction, epoch transition).

## Correctness checks

Every implementation must be compared against the sequential reference. Pass
`--dump-vector FILE` to any executable to dump the final normalized vector
outside the timed region, then diff the dumps for a strong, bitwise-style
check on small inputs:

```
./1-sequential/seq -n 5000 -nz 20000 -m irregular --dump-vector seq.dump
./4-mpi_omp/mpi_omp ... --dump-vector mpi.dump   # via mpirun
diff seq.dump mpi.dump
```

For larger inputs, correctness is verified by comparing the Rayleigh value
printed by each run against the sequential reference within a numerical
tolerance (default `1e-9`), since parallel floating-point reduction order is
not required to match the sequential one bitwise. This check is automated in
the scripts under `experiments_scripts/`.

## Reproducing the experiments

The scripts in `experiments_scripts/` run the executables on the spmcluster
and collect results into CSV files. `plot_results.py` regenerates the plots
included in the report. Each `run_*.py` script targets one experiment
(strong/weak scaling, speedup, scheduling comparison, MPI phase breakdown,
rank/thread interaction) and can be run independently. Before executing each script, a node (or up to 4 with MPI experiments) has to be reserved using the `salloc` command, but **the scripts are designed to be executed from the login node**, specifying the nodes using the command line parameter. See the header of
each script for its specific arguments.