#!/usr/bin/env python3

import argparse
from pathlib import Path

from utils_shared import (
    add_result,
    get_reference_rayleigh,
    run_repetitions,
    write_csv,
)

def run_time(executables, node, n, nz, mode, workers, repetitions, tolerance):
    rows = []
    cache = {}
    seq_results = run_repetitions(
        executables["seq"], node, n, nz, mode,
        repetitions=repetitions
    )
    reference = next(
        (r["rayleigh"] for r in seq_results if r["rayleigh"] is not None),
        None
    )
    if reference is None:
        print("Experiment failed: unable to obtain sequential Rayleigh reference")
        raise SystemExit(1)
    cache[(n, nz, mode)] = reference

    for name in ("seq", "thread", "openmp"):
        if name == "seq":
            results = seq_results
            threads = 1
            scheduling = ""
            chunk = 0
        else:
            threads = workers[-1]
            scheduling = "dynamic" if name == "thread" else ""
            chunk = 1024
            results = run_repetitions(
                executables[name], node, n, nz, mode,
                threads=threads,
                scheduling=scheduling or None,
                chunk=chunk,
                repetitions=repetitions,
            )

        for result in results:
            if name == "seq":
                error = None
                result_reference = None
            else:
                error = abs(result["rayleigh"] - reference)
                if error > tolerance:
                    print(
                        "Experiment failed due to insufficient result accuracy: "
                        f"|rayleigh - reference| = {error:.12e} > "
                        f"tolerance = {tolerance:.12e}"
                    )
                    raise SystemExit(1)
                result_reference = reference

            rows.append({
                "experiment": "time",
                "executable": name,
                "implementation": name,
                "n": n,
                "nz": nz,
                "mode": mode,
                "threads": threads,
                "scheduling": scheduling,
                "chunk_size": chunk,
                "repetition": result["repetition"],
                "time_s": result["time"],
                "rayleigh": result["rayleigh"],
                "reference_rayleigh": result_reference,
                "rayleigh_abs_error": error,
                "checksum": result["checksum"],
                "returncode": result["returncode"],
            })

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
        print(f"running execution time - {mode}")
        rows = run_time(
            executables, args.node, args.n, args.nz, mode,
            workers, args.repetitions, args.tolerance
        )
        write_csv(args.output / f"{mode}_time.csv", rows)

    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
