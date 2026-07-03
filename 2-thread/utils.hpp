#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "threadpool.hpp"

// Deterministic PRNG / mixing

// to avoid false sharing when changing values
struct alignas(64) padded_double { double value = 0.0; };
struct alignas(64) padded_uint64 { std::uint64_t value = 0; };

/** Struct to store the scheduling parameters: if the scheduling is dynamic and, in that case, the chunk size */
struct SchedulingParameters {
    bool is_dynamic = false;
    std::uint64_t chunk_size = 0;
};


/**
 * Function to get the static division of the N values of an array/matrix for the t-th thread out of the T threads
 */
std::pair<std::uint64_t, std::uint64_t> get_static_range(std::uint64_t N, std::uint64_t T, std::uint64_t t) {
    std::uint64_t base = N/T;
    std::uint64_t remainder = N%T;
    std::uint64_t start = t * base + std::min(t, remainder);
    std::uint64_t end = start + base + (t < remainder ? 1 : 0);
    return {start, end};
}





class SplitMix64 {
public:
    explicit SplitMix64(std::uint64_t seed) : state(seed) {}

    std::uint64_t next_u64() {
        state += 0x9e3779b97f4a7c15ULL;
        return mix(state);
    }

    double next_unit() {
        const std::uint64_t x = next_u64();
        return (x >> 11) * (1.0 / 9007199254740992.0);
    }

    static std::uint64_t mix(std::uint64_t x) {
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x = x ^ (x >> 31);
        return x;
    }

private:
    std::uint64_t state;
};

// Command-line parsing

static bool read_arg_u64(int argc, char** argv, const std::string& name, std::uint64_t& out) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) {
            out = std::strtoull(argv[i + 1], nullptr, 10);
            return true;
        }
    }
    return false;
}

static bool read_arg_str(int argc, char** argv, const std::string& name, std::string& out) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) {
            out = argv[i + 1];
            return true;
        }
    }
    return false;
}

static std::uint64_t checksum_vector(const std::vector<double>& x, ThreadPool &tp) {
    std::uint64_t checksum = 0;
    std::vector<padded_uint64> partial_checksums(tp.num_threads);

    for (std::uint64_t t = 0; t < tp.num_threads; t++) {
        auto [start, end] = get_static_range(x.size(), tp.num_threads, t);

        tp.enqueue([&x, t, &partial_checksums, start, end] {
            std::uint64_t local_checksum = 0;
            for (std::uint64_t i = start; i < end; i++) {
                std::uint64_t bits = 0;
                std::memcpy(&bits, &x[i], sizeof(double));
                local_checksum ^= SplitMix64::mix(bits ^ SplitMix64::mix(i));
            }
            partial_checksums[t].value = local_checksum;
        });
    }
    tp.barrier();

    for (std::uint64_t t = 0; t < tp.num_threads; t++) {
        checksum ^= partial_checksums[t].value;
    }

    return checksum;
}

static void dump_vector(const std::string& path, const std::vector<double>& x) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("could not open vector dump file: " + path);
    }

    out << std::setprecision(17);
    for (const double v : x) {
        out << v << '\n';
    }

    if (!out) {
        throw std::runtime_error("could not write vector dump file: " + path);
    }
}


static void usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " -n N -nz K -m regular|irregular [-s seed] [--dump-vector FILE]\n\n"
        << "Parameters:\n"
        << "  -n   Matrix size, NxN\n"
        << "  -nz  Total number of nonzeros\n"
        << "  -m   Matrix mode: regular or irregular\n"
        << "  -t   Number of threads in the thread pool\n"
        << "  -s   Optional seed, default 111\n"
        << "  --dump-vector FILE\n"
        << "       Optional output file for the final normalized vector\n";
}
