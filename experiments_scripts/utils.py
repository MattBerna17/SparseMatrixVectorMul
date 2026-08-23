import csv
import re
import subprocess

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RAYLEIGH_RE = re.compile(rf"rayleigh\s*=\s*(?P<value>{NUMBER})", re.IGNORECASE)
CHECKSUM_RE = re.compile(r"checksum\s*=\s*(?P<value>\S+)", re.IGNORECASE)
KEY_RE = re.compile(rf"^\s*(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>{NUMBER})\s*$", re.MULTILINE)
TIME_RE = re.compile(rf"Time\s*\(sec\)\s*=\s*(?P<value>{NUMBER})", re.IGNORECASE)

PHASES = [
    "computation",
    "global_spmv",
    "global_epoch",
    "global_comm",
    "global_rotate",
    "global_normalize",
]


def parse_output(output):
    timings = {}
    for match in KEY_RE.finditer(output):
        key = match.group("key").lower()
        if key in PHASES:
            timings[key] = float(match.group("value"))

    rayleigh = RAYLEIGH_RE.search(output)
    checksum = CHECKSUM_RE.search(output)

    return {
        "timings": timings,
        "rayleigh": float(rayleigh.group("value")) if rayleigh else None,
        "checksum": checksum.group("value") if checksum else None,
        "output": output,
    }


def run_program(executable, ranks, threads, n, nz, mode, chunk):
    command = [
        "mpirun", "-n", str(ranks), str(executable),
        "-n", str(n), "-nz", str(nz), "-m", mode, "-s", "111",
        "-t", str(threads), "--chunk-size", str(chunk),
    ]

    print(" ".join(command), flush=True)
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    parsed = parse_output(result.stdout)
    parsed["returncode"] = result.returncode
    return parsed


def repetitions(executable, ranks, threads, n, nz, mode, chunk, count,
                reference_rayleigh=None, tolerance=1e-9):
    rows = []
    for repetition in range(1, count + 1):
        result = run_program(executable, ranks, threads, n, nz, mode, chunk)
        check_rayleigh(result["rayleigh"], reference_rayleigh, tolerance)
        rows.append((repetition, result))
    return rows


def run_sequential(seq_executable, node, n, nz, mode, count=1):
    results = []
    for repetition in range(1, count + 1):
        command = [
            "srun", f"--nodelist={node}", str(seq_executable),
            "-n", str(n), "-nz", str(nz), "-m", mode, "-s", "111",
        ]

        print(" ".join(command), flush=True)
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        rayleigh = RAYLEIGH_RE.search(result.stdout)
        time = TIME_RE.search(result.stdout)

        results.append({
            "returncode": result.returncode,
            "rayleigh": float(rayleigh.group("value")) if rayleigh else None,
            "time": float(time.group("value")) if time else None,
            "output": result.stdout,
        })

    return results


def get_reference_rayleigh(seq_executable, node, n, nz, mode, cache):
    key = (n, nz, mode)
    if key not in cache:
        result = run_sequential(seq_executable, node, n, nz, mode, 1)[0]

        if result["returncode"] != 0 or result["rayleigh"] is None:
            print("Experiment failed: unable to obtain sequential Rayleigh reference")
            raise SystemExit(1)

        cache[key] = result["rayleigh"]

    return cache[key]


def check_rayleigh(rayleigh, reference_rayleigh, tolerance):
    if rayleigh is None:
        print("Experiment failed: Rayleigh value was not found in MPI+OpenMP output")
        raise SystemExit(1)

    error = abs(rayleigh - reference_rayleigh)

    if error > tolerance:
        print(
            "Experiment failed due to insufficient result accuracy: "
            f"|rayleigh - reference| = {error:.12e} > "
            f"tolerance = {tolerance:.12e}"
        )
        raise SystemExit(1)


def write_csv(path, rows):
    if not rows:
        return

    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_time_row(rows, experiment, executable, ranks, threads, n, nz, mode,
                 repetition, result, reference_rayleigh, tolerance):
    computation = result["timings"].get("computation")
    check_rayleigh(result["rayleigh"], reference_rayleigh, tolerance)
    rows.append({
        "experiment": experiment,
        "executable": executable.name,
        "ranks": ranks,
        "threads": threads,
        "n": n,
        "nz": nz,
        "mode": mode,
        "chunk_size": 1024,
        "repetition": repetition,
        "time_s": computation,
        "rayleigh": result["rayleigh"],
        "reference_rayleigh": reference_rayleigh,
        "rayleigh_error": abs(result["rayleigh"] - reference_rayleigh),
        "checksum": result["checksum"],
        "returncode": result["returncode"],
    })
