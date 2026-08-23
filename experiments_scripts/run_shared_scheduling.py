#!/usr/bin/env python3

import argparse
from pathlib import Path

from utils_shared import (
    add_result,
    get_reference_rayleigh,
    run_repetitions,
    write_csv,
)

def run_scheduling(executables, node, n, nz, mode, threads, chunk_sizes,
                   repetitions, tolerance):
    rows = []
    cache = {}
    reference = get_reference_rayleigh(
        executables["seq"], node, n, nz, mode, cache
    )

    for scheduling in ("static", "dynamic"):
        for chunk in chunk_sizes:
            for result in run_repetitions(
                executables["thread"], node, n, nz, mode,
                threads=threads,
                scheduling=scheduling,
                chunk=chunk,
                repetitions=repetitions,
            ):
                add_result(
                    rows, "thread_scheduling", executables["thread"], "thread",
                    n, nz, mode, threads, result["repetition"], result,
                    reference, tolerance, scheduling, chunk
                )

    for chunk in chunk_sizes:
        for result in run_repetitions(
            executables["openmp"], node, n, nz, mode,
            threads=threads,
            chunk=chunk,
            repetitions=repetitions,
        ):
            add_result(
                rows, "openmp_chunk", executables["openmp"], "openmp",
                n, nz, mode, threads, result["repetition"], result,
                reference, tolerance, "", chunk
            )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, type=Path)
    parser.add_argument("--thread", required=True, type=Path)
    parser.add_argument("--openmp", required=True, type=Path)
    parser.add_argument("--node", required=True)
    parser.add_argument("--n", type=int, default=500000)
    parser.add_argument("--nz", type=int, default=20000000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunk-sizes", default="64,256,1024,4096")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, default=Path("results_shared"))
    args = parser.parse_args()

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    executables = {"seq": args.seq, "thread": args.thread, "openmp": args.openmp}
    args.output.mkdir(parents=True, exist_ok=True)

    for mode in ("regular", "irregular"):
        print(f"running thread scheduling and openmp chunks - {mode}")
        rows = run_scheduling(
            executables, args.node, args.n, args.nz, mode,
            args.threads, chunk_sizes, args.repetitions, args.tolerance
        )

        write_csv(
            args.output / f"{mode}_thread_scheduling.csv",
            [r for r in rows if r["experiment"] == "thread_scheduling"]
        )
        write_csv(
            args.output / f"{mode}_openmp_chunks.csv",
            [r for r in rows if r["experiment"] == "openmp_chunk"]
        )

    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
