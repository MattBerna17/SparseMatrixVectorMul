CXX = g++
MPICXX = mpic++

CXXFLAGS = -O3 -std=c++20 -I . -Wall
OMPFLAGS = -fopenmp

SEQ = 1-sequential/seq
THREAD = 2-thread/thread
OPENMP = 3-openmp/omp
MPI_OMP = 4-mpi_omp/mpi_omp

.PHONY: all clean

all: clean
	$(CXX) $(CXXFLAGS) 1-sequential/iterative_SpMV.cpp -o $(SEQ)
	$(CXX) $(CXXFLAGS) 2-thread/thread_SpMV.cpp -o $(THREAD)
	$(CXX) $(CXXFLAGS) 3-openmp/openmp_SpMV.cpp -o $(OPENMP) $(OMPFLAGS)
	$(MPICXX) $(CXXFLAGS) 4-mpi_omp/mpi_omp_SpMV.cpp -o $(MPI_OMP) $(OMPFLAGS)

clean:
	rm -f $(SEQ) $(THREAD) $(OPENMP) $(MPI_OMP)