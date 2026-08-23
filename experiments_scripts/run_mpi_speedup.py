#!/usr/bin/env python3

import argparse
from pathlib import Path

from utils import (
    PHASES,
    add_time_row,
    check_rayleigh,
    get_reference_rayleigh,
    repetitions,
    run_sequential,
    write_csv,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi-omp", required=True, type=Path)
    parser.add_argument("--seq", required=True, type=Path)
    parser.add_argument("--node", required=True)
    parser.add_argument("--n", type=int, default=500000)
    parser.add_argument("--nz", type=int, default=20000000)
    parser.add_argument("--weak-n-per-rank", type=int, default=125000)
    parser.add_argument("--weak-nz-per-rank", type=int, default=5000000)
    parser.add_argument("--ranks", default="1,2,4")
    parser.add_argument("--threads", default="2,4")
    parser.add_argument("--strong-threads", type=int, default=4)
    parser.add_argument("--weak-threads", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, default=Path("results_mpi"))
    args = parser.parse_args()
    ranks = [int(x) for x in args.ranks.split(",")]
    threads = [int(x) for x in args.threads.split(",")]
    if any(x > 4 for x in ranks):
        parser.error("use at most 4 MPI ranks")
    args.output.mkdir(parents=True, exist_ok=True)
    return args, ranks, threads

def run_speedup(executable, seq_executable, node, n, nz, mode, ranks_list, threads, repetitions_count, tolerance):
    seq_results = run_sequential(seq_executable, node, n, nz, mode, repetitions_count)
    if any(result["returncode"] != 0 or result["rayleigh"] is None for result in seq_results):
        print("Experiment failed: unable to obtain sequential reference")
        raise SystemExit(1)

    reference_rayleigh = seq_results[0]["rayleigh"]
    for result in seq_results:
        if abs(result["rayleigh"] - reference_rayleigh) > tolerance:
            print("Experiment failed: sequential Rayleigh reference is not stable")
            raise SystemExit(1)
    check_rayleigh(result["rayleigh"], reference_rayleigh, tolerance)
    seq_times = [result["time"] for result in seq_results if result["time"] is not None]
    reference_time = sum(seq_times) / len(seq_times) if seq_times else None

    rows = []
    for ranks in ranks_list:
        for repetition, result in repetitions(
            executable, ranks, threads, n, nz, mode, 1024, repetitions_count,
            reference_rayleigh, tolerance):
            check_rayleigh(result["rayleigh"], reference_rayleigh, tolerance)
            time = result["timings"].get("computation")
            rows.append({
                "experiment": "speedup",
                "executable": executable.name,
                "ranks": ranks,
                "threads": threads,
                "n": n,
                "nz": nz,
                "mode": mode,
                "repetition": repetition,
                "time_s": time,
                "sequential_time_s": reference_time,
                "speedup": reference_time / time if reference_time and time else None,
                "rayleigh": result["rayleigh"],
                "reference_rayleigh": reference_rayleigh,
                "rayleigh_error": abs(result["rayleigh"] - reference_rayleigh),
                "returncode": result["returncode"],
            })
    return rows


def main():
    args, ranks, threads = parse_args()
    for mode in ["regular", "irregular"]:
        print("running mpi speedup")
        write_csv(args.output / f"{mode}_speedup.csv", run_speedup(
            args.mpi_omp, args.seq, args.node, args.n, args.nz, mode, ranks,
            args.strong_threads, args.repetitions, args.tolerance))
    print(f"results written to {args.output}")

if __name__ == "__main__":
    main()
