#!/usr/bin/env python3
"""Run and analyse the One Shot Project SPMV experiments on Slurm.

The script intentionally uses only the application's known CLI arguments:
  -n, -nz, -m, -t, and --chunk-size.
It invokes the hybrid executable through the Slurm form requested for the
project: ``srun --nodelist=<node-list> -n <MPI-ranks> <executable> ...``.

Typical use (inside an allocation obtained with salloc):
  python3 run_spmv_experiments.py \
    --seq ./sequential_SpMV --hybrid ./mpi_omp_SpMV \
    --nodes node01,node02 --ranks 1,2,4 --threads 1,2,4 \
    --strong-n 500000 --strong-nz 20000000 --matrix regular

For a manual allocation, use for example:
  salloc -w node01 --time=02:00:00
  ssh node01
  python3 ...
  scancel $SLURM_JOBID

The program output is retained verbatim.  Timing lines are discovered rather
than hard-coded, so newly added timings (e.g. communication or reduction
time) are automatically collected and plotted when their labels contain
"time", "tempo", "duration", or "elapsed".
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIME_RE = re.compile(
    rf"^\s*(?P<label>[^:=]+?(?:time|tempo|duration|elapsed)[^:=]*)\s*[:=]\s*"
    rf"(?P<value>{NUMBER})\s*(?P<unit>ns|us|µs|ms|s|sec|seconds)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
RAYLEIGH_RE = re.compile(rf"\brayleigh\b[^:=]*[:=]\s*(?P<value>{NUMBER})", re.I)
CHECKSUM_RE = re.compile(r"\bchecksum\b[^:=]*[:=]\s*(?P<value>[^\s,;]+)", re.I)


def csv_ints(value: str) -> list[int]:
    """Parse a non-empty comma-separated integer list."""
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("use a non-empty list of positive integers")
    return result


def normalise_label(label: str) -> str:
    """Turn a printed timing label into a stable CSV/JSON column name."""
    label = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"time_{label}" if not label.startswith("time_") else label


def seconds(value: float, unit: str | None) -> float:
    scales = {None: 1.0, "s": 1.0, "sec": 1.0, "seconds": 1.0,
              "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9}
    return value * scales[unit.lower() if unit else None]


def parse_output(text: str) -> tuple[dict[str, float], float | None, str | None]:
    """Extract all available timings and the two numerical diagnostics."""
    timings: dict[str, float] = {}
    for match in TIME_RE.finditer(text):
        timings[normalise_label(match.group("label"))] = seconds(
            float(match.group("value")), match.group("unit")
        )
    rayleigh = RAYLEIGH_RE.search(text)
    checksum = CHECKSUM_RE.search(text)
    return timings, (float(rayleigh.group("value")) if rayleigh else None), (
        checksum.group("value") if checksum else None
    )


def choose_total_time(timings: dict[str, float]) -> float | None:
    """Prefer an explicit total time; otherwise use the first collected time."""
    priority = ("total", "execution", "overall", "wall")
    for word in priority:
        for name, value in timings.items():
            if word in name:
                return value
    return next(iter(timings.values()), None)


def close_enough(reference: float | None, observed: float | None,
                 rel_tol: float, abs_tol: float) -> tuple[bool | None, float | None]:
    if reference is None or observed is None:
        return None, None
    difference = abs(reference - observed)
    return math.isclose(reference, observed, rel_tol=rel_tol, abs_tol=abs_tol), difference


def checksum_comparison(reference: str | None, observed: str | None,
                        rel_tol: float, abs_tol: float) -> tuple[bool | None, float | None]:
    """Compare numeric checksums approximately; opaque hash checksums exactly.

    A bitwise/XOR checksum is deliberately not expected to remain identical
    after parallel reductions; the CSV records it, but does not falsely claim
    a numerical tolerance for an opaque integer/hash value.
    """
    if reference is None or observed is None:
        return None, None
    try:
        ref, got = float(reference), float(observed)
    except ValueError:
        return reference == observed, None
    return close_enough(ref, got, rel_tol, abs_tol)


@dataclasses.dataclass(frozen=True)
class RunSpec:
    experiment: str
    n: int
    nz: int
    ranks: int
    threads: int
    repetition: int


class ExperimentRunner:
    def __init__(self, args: argparse.Namespace, run_dir: Path) -> None:
        self.args = args
        self.run_dir = run_dir
        self.logs_dir = run_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.seq_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def command(self, executable: Path, spec: RunSpec, hybrid: bool) -> list[str]:
        # The sequential reference receives only problem-definition flags.  In
        # particular, do not assume it accepts the hybrid-only -t or
        # --chunk-size options.
        app = [str(executable), "-n", str(spec.n), "-nz", str(spec.nz),
               "-m", self.args.matrix]
        if not hybrid:
            return app
        app += ["-t", str(spec.threads), "--chunk-size", str(self.args.chunk_size)]
        return ["srun", f"--nodelist={self.args.nodes}", "-n", str(spec.ranks), *app]

    def execute(self, executable: Path, spec: RunSpec, hybrid: bool) -> dict[str, Any]:
        cmd = self.command(executable, spec, hybrid)
        tag = (f"{spec.experiment}_n{spec.n}_nz{spec.nz}_p{spec.ranks}_"
               f"t{spec.threads}_r{spec.repetition}_{'hybrid' if hybrid else 'seq'}")
        log_path = self.logs_dir / f"{tag}.log"
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        print("Running:", " ".join(cmd), flush=True)
        try:
            completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT,
                                       timeout=self.args.timeout, check=False)
            output, returncode = completed.stdout, completed.returncode
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or "") + "\nTIMEOUT\n"
            returncode = 124
        log_path.write_text(output, encoding="utf-8")
        timings, rayleigh, checksum = parse_output(output)
        row: dict[str, Any] = {
            **dataclasses.asdict(spec), "implementation": "hybrid" if hybrid else "sequential",
            "command": " ".join(cmd), "started_utc": started, "returncode": returncode,
            "log_file": str(log_path), "rayleigh": rayleigh, "checksum": checksum,
            "total_time_s": choose_total_time(timings), **timings,
        }
        self.rows.append(row)
        if returncode != 0:
            print(f"WARNING: failed run (exit {returncode}); see {log_path}", file=sys.stderr)
        return row

    def sequential(self, n: int, nz: int, repetition: int) -> dict[str, Any]:
        key = (n, nz)
        if key not in self.seq_cache:
            spec = RunSpec("baseline", n, nz, 1, 1, repetition)
            self.seq_cache[key] = self.execute(self.args.seq, spec, hybrid=False)
        return self.seq_cache[key]

    def hybrid(self, spec: RunSpec) -> None:
        baseline = self.sequential(spec.n, spec.nz, spec.repetition)
        row = self.execute(self.args.hybrid, spec, hybrid=True)
        ray_ok, ray_delta = close_enough(baseline["rayleigh"], row["rayleigh"],
                                         self.args.rel_tol, self.args.abs_tol)
        chk_ok, chk_delta = checksum_comparison(baseline["checksum"], row["checksum"],
                                                self.args.rel_tol, self.args.abs_tol)
        row.update({
            "reference_rayleigh": baseline["rayleigh"], "rayleigh_match": ray_ok,
            "rayleigh_abs_difference": ray_delta, "reference_checksum": baseline["checksum"],
            "checksum_match": chk_ok, "checksum_abs_difference": chk_delta,
            "speedup_vs_sequential": (
                baseline["total_time_s"] / row["total_time_s"]
                if baseline["total_time_s"] and row["total_time_s"] else None
            ),
        })
        if ray_ok is False or chk_ok is False:
            print("WARNING: numerical check outside the configured tolerance:",
                  f"Rayleigh={ray_ok}, checksum={chk_ok}", file=sys.stderr)

    def run(self) -> None:
        if self.args.check_queue:
            subprocess.run(["squeue"], check=False)
        # Strong scaling: one fixed global problem, varying ranks and threads.
        for rep in range(1, self.args.repetitions + 1):
            for ranks in self.args.ranks:
                for threads in self.args.threads:
                    self.hybrid(RunSpec("strong", self.args.strong_n, self.args.strong_nz,
                                        ranks, threads, rep))
        # Weak scaling: n and nnz both grow linearly with MPI rank count.
        for rep in range(1, self.args.repetitions + 1):
            for ranks in self.args.ranks:
                for threads in self.args.threads:
                    self.hybrid(RunSpec("weak", self.args.weak_n_per_rank * ranks,
                                        self.args.weak_nz_per_rank * ranks,
                                        ranks, threads, rep))


def write_table(rows: list[dict[str, Any]], destination: Path) -> None:
    keys = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Create aggregate scaling plots plus one plot per detected timing field."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable: CSV/JSON written but plots skipped.", file=sys.stderr)
        return
    hybrid = [r for r in rows if r.get("implementation") == "hybrid" and
              r.get("returncode") == 0 and r.get("total_time_s") is not None]
    if not hybrid:
        return
    timing_columns = sorted({key for row in hybrid for key in row if key.startswith("time_")})
    metrics = ["total_time_s", "speedup_vs_sequential", *timing_columns]

    for experiment in ("strong", "weak"):
        data = [r for r in hybrid if r["experiment"] == experiment]
        for metric in metrics:
            usable = [r for r in data if isinstance(r.get(metric), (int, float))]
            if not usable:
                continue
            plt.figure(figsize=(8, 5))
            for threads in sorted({r["threads"] for r in usable}):
                points = [r for r in usable if r["threads"] == threads]
                grouped: dict[int, list[float]] = {}
                for row in points:
                    grouped.setdefault(row["ranks"], []).append(float(row[metric]))
                xs = sorted(grouped)
                ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
                plt.plot(xs, ys, marker="o", label=f"OMP threads={threads}")
            plt.xlabel("MPI ranks")
            plt.ylabel(metric.replace("_", " ") + " (s)" if metric != "speedup_vs_sequential"
                       else "speedup vs sequential")
            plt.title(f"{experiment.capitalize()} scaling - {metric}")
            plt.grid(True, alpha=.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{experiment}_{metric}.png", dpi=160)
            plt.close()

    # Explicit MPI x OMP sweep heatmaps for total time and all component times.
    sweep = [r for r in hybrid if r["experiment"] == "strong"]
    for metric in ["total_time_s", *timing_columns]:
        values = {(r["ranks"], r["threads"]): [] for r in sweep}
        for row in sweep:
            if isinstance(row.get(metric), (int, float)):
                values[(row["ranks"], row["threads"])].append(float(row[metric]))
        ranks, threads = sorted({r[0] for r in values}), sorted({r[1] for r in values})
        if not ranks or not threads:
            continue
        matrix = [[(sum(values.get((p, t), [])) / len(values[(p, t)]))
                   if values.get((p, t)) else float("nan") for t in threads] for p in ranks]
        plt.figure(figsize=(7, 5))
        image = plt.imshow(matrix, aspect="auto", origin="lower")
        plt.colorbar(image, label=f"{metric} (s)")
        plt.xticks(range(len(threads)), threads)
        plt.yticks(range(len(ranks)), ranks)
        plt.xlabel("OpenMP threads")
        plt.ylabel("MPI ranks")
        plt.title(f"Strong-scaling MPI x OpenMP sweep - {metric}")
        plt.tight_layout()
        plt.savefig(output_dir / f"sweep_heatmap_{metric}.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", required=True, type=Path, help="sequential executable")
    parser.add_argument("--hybrid", required=True, type=Path, help="MPI+OpenMP executable")
    parser.add_argument("--nodes", required=True, help="Slurm node or comma-separated node list")
    parser.add_argument("--ranks", type=csv_ints, default=[1, 2, 4], help="MPI rank sweep")
    parser.add_argument("--threads", type=csv_ints, default=[1, 2, 4], help="OpenMP thread sweep")
    parser.add_argument("--strong-n", type=int, required=True, help="fixed N for strong scaling")
    parser.add_argument("--strong-nz", type=int, required=True, help="fixed NNZ for strong scaling")
    parser.add_argument("--weak-n-per-rank", type=int, default=None, help="N contributed by each rank")
    parser.add_argument("--weak-nz-per-rank", type=int, default=None, help="NNZ contributed by each rank")
    parser.add_argument("--matrix", choices=("regular", "irregular"), default="regular")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=None, help="per-run timeout in seconds")
    parser.add_argument("--rel-tol", type=float, default=1e-7, help="relative numerical tolerance")
    parser.add_argument("--abs-tol", type=float, default=1e-10, help="absolute numerical tolerance")
    parser.add_argument("--check-queue", action="store_true", help="print squeue before the campaign")
    parser.add_argument("--output", type=Path, default=Path("spmv_results"))
    args = parser.parse_args()
    if args.weak_n_per_rank is None:
        args.weak_n_per_rank = args.strong_n
    if args.weak_nz_per_rank is None:
        args.weak_nz_per_rank = args.strong_nz
    for executable in (args.seq, args.hybrid):
        if not executable.is_file():
            parser.error(f"executable not found: {executable}")
    return args


def main() -> int:
    args = parse_args()
    if shutil.which("srun") is None:
        print("ERROR: srun is not available in PATH. Run this on the Slurm cluster.", file=sys.stderr)
        return 2
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output / stamp
    runner = ExperimentRunner(args, run_dir)
    runner.run()
    write_table(runner.rows, run_dir / "results.csv")
    (run_dir / "results.json").write_text(json.dumps(runner.rows, indent=2, default=str), encoding="utf-8")
    plot(runner.rows, run_dir)
    failures = sum(row.get("returncode", 0) != 0 for row in runner.rows)
    checks = [row for row in runner.rows if row.get("implementation") == "hybrid"]
    bad = sum(row.get("rayleigh_match") is False or row.get("checksum_match") is False for row in checks)
    print(f"Finished. Results: {run_dir} | failed runs: {failures} | numerical warnings: {bad}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
