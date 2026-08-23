#!/usr/bin/env python3

import argparse
from pathlib import Path

from utils_shared import (
    add_result,
    get_reference_rayleigh,
    run_repetitions,
    write_csv,
)


def run_sweep(executables, node, n, nz, mode, threads, chunk_sizes,
              repetitions, tolerance, cache):
    rows = []
    reference = get_reference_rayleigh(
        executables["seq"], node, n, nz, mode, cache
    )

    for chunk in chunk_sizes:
        for result in run_repetitions(
            executables["task"], node, n, nz, mode,
            threads=threads,
            chunk=chunk,
            repetitions=repetitions,
        ):
            add_result(
                rows, "task_vs_worksharing", executables["task"], "task-based",
                n, nz, mode, threads, result["repetition"], result,
                reference, tolerance, "", chunk
            )

        for result in run_repetitions(
            executables["worksharing"], node, n, nz, mode,
            threads=threads,
            chunk=chunk,
            repetitions=repetitions,
        ):
            add_result(
                rows, "task_vs_worksharing",
                executables["worksharing"], "work-sharing",
                n, nz, mode, threads, result["repetition"], result,
                reference, tolerance, "", chunk
            )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, type=Path)
    parser.add_argument("--omp-task", required=True, type=Path)
    parser.add_argument("--omp-worksharing", required=True, type=Path)
    parser.add_argument("--node", required=True)
    parser.add_argument("--n", type=int, default=500000)
    parser.add_argument("--nz", type=int, default=20000000)
    parser.add_argument("--threads", default="1,2,4")
    parser.add_argument("--chunk-sizes", default="256,1024,4096")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--output", type=Path, default=Path("results_openmp_comparison")
    )
    args = parser.parse_args()

    threads_values = [int(x) for x in args.threads.split(",")]
    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    executables = {
        "seq": args.seq,
        "task": args.omp_task,
        "worksharing": args.omp_worksharing,
    }
    args.output.mkdir(parents=True, exist_ok=True)

    mode = "regular"
    print(f"running OpenMP task vs work-sharing - {mode}")

    rows = []
    cache = {}
    for threads in threads_values:
        rows.extend(run_sweep(
            executables, args.node, args.n, args.nz, mode,
            threads, chunk_sizes, args.repetitions, args.tolerance, cache
        ))

    write_csv(
        args.output / f"{mode}_openmp_task_vs_worksharing.csv",
        rows
    )

    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
