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

def run_phases(executable, seq_executable, node, n, nz, mode, ranks, threads, repetitions_count, tolerance):
    rows = []
    cache = {}
    reference = get_reference_rayleigh(seq_executable, node, n, nz, mode, cache)
    for rank in ranks:
        for thread in threads:
            for repetition, result in repetitions(
                executable, rank, thread, n, nz, mode, 1024, repetitions_count,
                reference, tolerance):
                check_rayleigh(result["rayleigh"], reference, tolerance)
                for phase in PHASES:
                    rows.append({
                        "experiment": "phases",
                        "executable": executable.name,
                        "ranks": rank,
                        "threads": thread,
                        "n": n,
                        "nz": nz,
                        "mode": mode,
                        "phase": phase,
                        "repetition": repetition,
                        "time_s": result["timings"].get(phase),
                        "rayleigh": result["rayleigh"],
                        "reference_rayleigh": reference,
                        "rayleigh_error": abs(result["rayleigh"] - reference),
                        "returncode": result["returncode"],
                    })
    return rows


def main():
    args, ranks, threads = parse_args()
    for mode in ["regular", "irregular"]:
        print("running mpi phases")
        write_csv(args.output / f"{mode}_phases.csv", run_phases(
            args.mpi_omp, args.seq, args.node, args.n, args.nz, mode, ranks,
            threads, args.repetitions, args.tolerance))
    print(f"results written to {args.output}")

if __name__ == "__main__":
    main()
