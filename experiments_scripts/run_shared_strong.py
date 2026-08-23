#!/usr/bin/env python3

import argparse
from pathlib import Path

from utils_shared import (
    add_result,
    get_reference_rayleigh,
    run_repetitions,
    write_csv,
)

def run_strong(executables, node, n, nz, mode, workers, repetitions, tolerance):
    rows = []
    cache = {}
    reference = get_reference_rayleigh(
        executables["seq"], node, n, nz, mode, cache
    )

    seq = run_repetitions(executables["seq"], node, n, nz, mode, repetitions=repetitions)
    for result in seq:
            add_result(rows, "strong", executables["seq"], "seq", n, nz, mode,
                       1, result["repetition"], result, reference_rayleigh=reference)

    for name in ("thread", "openmp"):
        for threads in workers:
            scheduling = "dynamic" if name == "thread" else ""

            for result in run_repetitions(
                executables[name], node, n, nz, mode,
                threads=threads,
                scheduling=scheduling,
                chunk=1024,
                repetitions=repetitions,
            ):
                add_result(
                    rows, "strong", executables[name], name,
                    n, nz, mode, threads, result["repetition"], result,
                    reference, tolerance, scheduling, 1024
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
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output", type=Path, default=Path("results_shared"))
    args = parser.parse_args()

    workers = [int(x) for x in args.workers.split(",")]
    executables = {"seq": args.seq, "thread": args.thread, "openmp": args.openmp}
    args.output.mkdir(parents=True, exist_ok=True)

    for mode in ("regular", "irregular"):
        print(f"running strong scalability - {mode}")
        rows = run_strong(
            executables, args.node, args.n, args.nz, mode,
            workers, args.repetitions, args.tolerance
        )
        write_csv(args.output / f"{mode}_strong.csv", rows)

    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
