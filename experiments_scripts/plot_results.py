#!/usr/bin/env python3
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def load(path):
    return pd.read_csv(path) if path.exists() else None


def error_data(data, x):
    grouped = data.groupby(x)["time_s"]
    mean = grouped.mean()
    low = mean - grouped.min()
    high = grouped.max() - mean
    return mean, low, high


def plot_strong(data, out, mode):
    if data is None or data.empty:
        return
    plt.figure(figsize=(10, 6))
    for name in ["seq", "thread", "openmp"]:
        d = data[data["implementation"] == name]
        if d.empty:
            continue
        mean, low, high = error_data(d, "threads")
        plt.errorbar(mean.index, mean.values, yerr=[low, high],
                     marker="o", capsize=4, linewidth=2, label=name)
    n, nz = int(data.n.iloc[0]), int(data.nz.iloc[0])
    plt.title(f"strong scalability - {mode} workload - N = {n:,}, nnz = {nz:,} - execution time")
    plt.xlabel("number of threads")
    plt.ylabel("execution time [s]")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{mode}_strong.png", dpi=200)
    plt.close()


def plot_weak(data, out, mode):
    if data is None or data.empty:
        return
    plt.figure(figsize=(10, 6))
    for name in ["seq", "thread", "openmp"]:
        d = data[data["implementation"] == name]
        if d.empty:
            continue
        mean, low, high = error_data(d, "threads")
        plt.errorbar(mean.index, mean.values, yerr=[low, high],
                     marker="o", capsize=4, linewidth=2, label=name)
    n = int(data.n.iloc[0] / data.threads.iloc[0])
    nz = int(data.nz.iloc[0] / data.threads.iloc[0])
    plt.title(f"weak scalability - {mode} workload - N/thread = {n:,}, nnz/thread = {nz:,} - execution time")
    plt.xlabel("number of threads")
    plt.ylabel("execution time [s]")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{mode}_weak.png", dpi=200)
    plt.close()


def plot_speedup(data, out, mode):
    if data is None or data.empty or "speedup" not in data:
        return
    plt.figure(figsize=(10, 6))
    for name in ["thread", "openmp"]:
        d = data[data["implementation"] == name]
        if d.empty:
            continue
        g = d.groupby("threads")["speedup"]
        mean, low, high = g.mean(), g.mean()-g.min(), g.max()-g.mean()
        plt.errorbar(mean.index, mean.values, yerr=[low, high],
                     marker="o", capsize=4, linewidth=2, label=name)
    n, nz = int(data.n.iloc[0]), int(data.nz.iloc[0])
    plt.title(f"speedup vs sequential - {mode} workload - N = {n:,}, nnz = {nz:,}")
    plt.xlabel("number of threads")
    plt.ylabel("speedup")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{mode}_speedup.png", dpi=200)
    plt.close()


def plot_time(data, out, mode):
    if data is None or data.empty:
        return
    g = data.groupby("implementation")["time_s"]
    mean, low, high = g.mean(), g.mean()-g.min(), g.max()-g.mean()
    plt.figure(figsize=(10, 6))
    plt.bar(mean.index, mean.values, yerr=[low, high], capsize=5)
    n, nz = int(data.n.iloc[0]), int(data.nz.iloc[0])
    plt.title(f"execution time - {mode} workload - N = {n:,}, nnz = {nz:,}")
    plt.xlabel("implementation")
    plt.ylabel("execution time [s]")
    plt.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(out / f"{mode}_time.png", dpi=200)
    plt.close()


def plot_thread_scheduling(data, out, mode):
    if data is None or data.empty:
        return
    plt.figure(figsize=(10, 6))
    for scheduling in ["static", "dynamic"]:
        d = data[data.scheduling == scheduling]
        if d.empty:
            continue
        mean, low, high = error_data(d, "chunk_size")
        plt.errorbar(mean.index, mean.values, yerr=[low, high],
                     marker="o", capsize=4, linewidth=2, label=scheduling)
    n, nz, t = int(data.n.iloc[0]), int(data.nz.iloc[0]), int(data.threads.iloc[0])
    plt.title(f"ThreadPool static vs dynamic - {mode} workload - N = {n:,}, nnz = {nz:,}, threads = {t}")
    plt.xlabel("chunk size")
    plt.ylabel("execution time [s]")
    plt.xscale("log", base=2)
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{mode}_thread_scheduling.png", dpi=200)
    plt.close()


def plot_openmp_chunks(data, out, mode):
    if data is None or data.empty:
        return
    mean, low, high = error_data(data, "chunk_size")
    plt.figure(figsize=(10, 6))
    plt.errorbar(mean.index, mean.values, yerr=[low, high],
                 marker="o", capsize=4, linewidth=2)
    n, nz, t = int(data.n.iloc[0]), int(data.nz.iloc[0]), int(data.threads.iloc[0])
    plt.title(f"OpenMP chunk-size sweep - {mode} workload - N = {n:,}, nnz = {nz:,}, threads = {t}")
    plt.xlabel("chunk size")
    plt.ylabel("execution time [s]")
    plt.xscale("log", base=2)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out / f"{mode}_openmp_chunks.png", dpi=200)
    plt.close()


def plot_modes(input_dir, out):
    regular = load(input_dir / "regular_strong.csv")
    irregular = load(input_dir / "irregular_strong.csv")
    if regular is None or irregular is None:
        return
    plt.figure(figsize=(10, 6))
    for name in ["seq", "thread", "openmp"]:
        for mode, data, style in [
            ("regular", regular, "-"),
            ("irregular", irregular, "--"),
        ]:
            d = data[data.implementation == name]
            if not d.empty:
                mean = d.groupby("threads").time_s.mean()
                plt.plot(mean.index, mean.values, marker="o", linestyle=style,
                         linewidth=2, label=f"{name} - {mode}")
    n, nz = int(regular.n.iloc[0]), int(regular.nz.iloc[0])
    plt.title(f"regular vs irregular workload - strong scalability - N = {n:,}, nnz = {nz:,}")
    plt.xlabel("number of threads")
    plt.ylabel("execution time [s]")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "regular_vs_irregular.png", dpi=200)
    plt.close()


def plot_openmp_task_vs_worksharing(csv_file, out, mode):
    if not csv_file.exists():
        return
    data = pd.read_csv(csv_file)

    grouped = (
        data
        .groupby(["threads", "implementation"])["time_s"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for implementation in ["task-based", "work-sharing"]:
        current = grouped[grouped["implementation"] == implementation]
        if current.empty:
            continue

        plt.errorbar(
            current["threads"],
            current["mean"],
            yerr=[
                current["mean"] - current["min"],
                current["max"] - current["mean"]
            ],
            marker="o",
            capsize=4,
            label=implementation
        )

    n = data["n"].iloc[0]
    nz = data["nz"].iloc[0]

    plt.title(
        f"OpenMP task-based vs work-sharing - "
        f"mode = {mode}, N = {n}, nnz = {nz}"
    )
    plt.xlabel("Number of OpenMP threads")
    plt.ylabel("Execution time (s)")
    plt.xticks(sorted(data["threads"].unique()))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(out / f"{mode}_openmp_task_vs_worksharing.png", dpi=200)
    plt.close()


# ----------------------------------------------------------------------
# MPI+OpenMP plots
# ----------------------------------------------------------------------

def sequential_reference_time(speedup_data):
    # The only MPI CSV that carries a sequential baseline is *_speedup.csv
    # (column "sequential_time_s"), already averaged over repetitions by
    # the collection script. Returns None if unavailable.
    if speedup_data is None or speedup_data.empty or "sequential_time_s" not in speedup_data:
        return None
    return speedup_data.sequential_time_s.iloc[0]


def plot_mpi_strong(data, out, mode, seq_time=None):
    if data is None or data.empty:
        return
    plt.figure(figsize=(10, 6))
    for threads in sorted(data.threads.unique()):
        d = data[data.threads == threads]
        mean, low, high = error_data(d, "ranks")
        plt.errorbar(mean.index, mean.values, yerr=[low, high],
                     marker="o", capsize=4, linewidth=2, label=f"{int(threads)} OpenMP threads")
    if seq_time is not None:
        plt.axhline(seq_time, color="black", linestyle="--", linewidth=2, label="sequential")
    n, nz = int(data.n.iloc[0]), int(data.nz.iloc[0])
    plt.title(f"MPI+OpenMP strong scalability - {mode} workload - N = {n:,}, nnz = {nz:,} - execution time")
    plt.xlabel("number of MPI ranks")
    plt.ylabel("execution time [s]")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{mode}_mpi_strong.png", dpi=200)
    plt.close()


def plot_mpi_weak(data, out, mode, seq_time=None):
    if data is None or data.empty:
        return
    plt.figure(figsize=(10, 6))
    for threads in sorted(data.threads.unique()):
        d = data[data.threads == threads]
        mean, low, high = error_data(d, "ranks")
        plt.errorbar(mean.index, mean.values, yerr=[low, high],
                     marker="o", capsize=4, linewidth=2, label=f"{int(threads)} OpenMP threads")
    if seq_time is not None:
        plt.axhline(seq_time, color="black", linestyle="--", linewidth=2, label="sequential (base problem size)")
    smallest_ranks = data.ranks.min()
    baseline = data[data.ranks == smallest_ranks]
    n = int(baseline.n.iloc[0] / smallest_ranks)
    nz = int(baseline.nz.iloc[0] / smallest_ranks)
    plt.title(f"MPI+OpenMP weak scalability - {mode} workload - N/rank = {n:,}, nnz/rank = {nz:,} - execution time")
    plt.xlabel("number of MPI ranks")
    plt.ylabel("execution time [s]")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{mode}_mpi_weak.png", dpi=200)
    plt.close()


def plot_mpi_speedup(data, out, mode):
    if data is None or data.empty or "speedup" not in data:
        return
    plt.figure(figsize=(10, 6))
    for threads in sorted(data.threads.unique()):
        d = data[data.threads == threads]
        g = d.groupby("ranks")["speedup"]
        mean, low, high = g.mean(), g.mean()-g.min(), g.max()-g.mean()
        plt.errorbar(mean.index, mean.values, yerr=[low, high],
                     marker="o", capsize=4, linewidth=2, label=f"{int(threads)} OpenMP threads")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=2, label="sequential")
    n, nz = int(data.n.iloc[0]), int(data.nz.iloc[0])
    plt.title(f"MPI+OpenMP speedup vs sequential - {mode} workload - N = {n:,}, nnz = {nz:,}")
    plt.xlabel("number of MPI ranks")
    plt.ylabel("speedup")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{mode}_mpi_speedup.png", dpi=200)
    plt.close()


def plot_mpi_time(data, out, mode):
    if data is None or data.empty:
        return
    grouped = data.groupby(["ranks", "threads"])["time_s"]
    mean, low, high = grouped.mean(), grouped.mean()-grouped.min(), grouped.max()-grouped.mean()
    labels = [f"{int(r)}r x {int(t)}t" for r, t in mean.index]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, mean.values, yerr=[low.values, high.values], capsize=5)
    n, nz = int(data.n.iloc[0]), int(data.nz.iloc[0])
    plt.title(f"MPI+OpenMP execution time - {mode} workload - N = {n:,}, nnz = {nz:,}")
    plt.xlabel("MPI ranks x OpenMP threads")
    plt.ylabel("execution time [s]")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(out / f"{mode}_mpi_time.png", dpi=200)
    plt.close()


def plot_mpi_rank_thread(data, out, mode, seq_time=None):
    if data is None or data.empty:
        return
    plt.figure(figsize=(10, 6))
    for ranks in sorted(data.ranks.unique()):
        d = data[data.ranks == ranks]
        mean, low, high = error_data(d, "threads")
        plt.errorbar(mean.index, mean.values, yerr=[low, high],
                     marker="o", capsize=4, linewidth=2,
                     label=f"{int(ranks)} MPI ranks")
    if seq_time is not None:
        plt.axhline(seq_time, color="black", linestyle="--", linewidth=2, label="sequential")
    n, nz = int(data.n.iloc[0]), int(data.nz.iloc[0])
    plt.title(f"MPI+OpenMP rank/thread interaction - {mode} workload - N = {n:,}, nnz = {nz:,} - execution time")
    plt.xlabel("OpenMP threads per rank")
    plt.ylabel("execution time [s]")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{mode}_mpi_rank_thread_interaction.png", dpi=200)
    plt.close()


def plot_mpi_phase_breakdown(data, out, mode, seq_time=None):
    if data is None or data.empty:
        return
    # "computation" is the total wall-clock time of the timed region, not a
    # phase on top of the others (it is ~ the sum of the phases below), so
    # it is excluded from the stack to avoid double counting.
    phase_names = [x for x in [
        "global_spmv", "global_epoch",
        "global_comm", "global_rotate", "global_normalize"
    ] if x in data.phase.unique()]
    if not phase_names:
        return

    pivot = data.pivot_table(index=["ranks", "threads"], columns="phase",
                             values="time_s", aggfunc="mean")
    pivot = pivot[phase_names]
    labels = [f"{int(r)}r x {int(t)}t" for r, t in pivot.index]
    if seq_time is not None:
        labels = labels + ["sequential"]

    plt.figure(figsize=(12, 7))
    bottom = None
    for phase in phase_names:
        values = pivot[phase].values
        if seq_time is not None:
            # the sequential run has no phase breakdown, only a total time;
            # it is shown as a single reference bar with all its time
            # attributed to the first phase so the bar reaches seq_time.
            filler = seq_time if phase == phase_names[0] else 0.0
            values = list(values) + [filler]
        plt.bar(labels, values, bottom=bottom, label=phase)
        bottom = values if bottom is None else [b + v for b, v in zip(bottom, values)]
    n, nz = int(data.n.iloc[0]), int(data.nz.iloc[0])
    plt.title(f"MPI+OpenMP execution-time breakdown - {mode} workload - N = {n:,}, nnz = {nz:,}")
    plt.xlabel("MPI ranks x OpenMP threads")
    plt.ylabel("time [s]")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / f"{mode}_mpi_phase_breakdown.png", dpi=200)
    plt.close()


def plot_mpi(input_dir, out, mode):
    speedup_data = load(input_dir / f"{mode}_speedup.csv")
    seq_time = sequential_reference_time(speedup_data)

    plot_mpi_strong(load(input_dir / f"{mode}_strong.csv"), out, mode, seq_time)
    # weak.csv has no matching sequential baseline: the problem size grows
    # with the rank count, so the fixed-size sequential time from
    # speedup.csv is not a valid reference for every point on this curve.
    plot_mpi_weak(load(input_dir / f"{mode}_weak.csv"), out, mode)
    plot_mpi_speedup(speedup_data, out, mode)
    plot_mpi_time(load(input_dir / f"{mode}_time.csv"), out, mode)
    plot_mpi_rank_thread(load(input_dir / f"{mode}_rank_thread.csv"), out, mode, seq_time)
    plot_mpi_phase_breakdown(load(input_dir / f"{mode}_phases.csv"), out, mode, seq_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results_shared"))
    parser.add_argument("--openmp-comparison-input", type=Path, default=Path("results_openmp_comparison"))
    parser.add_argument("--mpi-input", type=Path, default=Path("results_mpi"))
    parser.add_argument("--output", type=Path, default=Path("plots"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for mode in ["regular", "irregular"]:
        plot_strong(load(args.input / f"{mode}_strong.csv"), args.output, mode)
        plot_weak(load(args.input / f"{mode}_weak.csv"), args.output, mode)
        plot_speedup(load(args.input / f"{mode}_speedup.csv"), args.output, mode)
        plot_time(load(args.input / f"{mode}_time.csv"), args.output, mode)
        plot_thread_scheduling(load(args.input / f"{mode}_thread_scheduling.csv"), args.output, mode)
        plot_openmp_chunks(load(args.input / f"{mode}_openmp_chunks.csv"), args.output, mode)
        plot_openmp_task_vs_worksharing(
            args.openmp_comparison_input / f"{mode}_openmp_task_vs_worksharing.csv",
            args.output, mode)
        plot_mpi(args.mpi_input, args.output, mode)

    plot_modes(args.input, args.output)
    print(f"plots written to {args.output}")


if __name__ == "__main__":
    main()