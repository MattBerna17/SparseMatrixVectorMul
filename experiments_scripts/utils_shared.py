import csv
import re
import subprocess

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIME_RE = re.compile(rf"Time\s*\(sec\)\s*=\s*(?P<value>{NUMBER})", re.IGNORECASE)
RAYLEIGH_RE = re.compile(rf"rayleigh\s*=\s*(?P<value>{NUMBER})", re.IGNORECASE)
CHECKSUM_RE = re.compile(r"checksum\s*=\s*(?P<value>\S+)", re.IGNORECASE)


def parse_output(output):
    time_match = TIME_RE.search(output)
    rayleigh_match = RAYLEIGH_RE.search(output)
    checksum_match = CHECKSUM_RE.search(output)

    return {
        "time": float(time_match.group("value")) if time_match else None,
        "rayleigh": float(rayleigh_match.group("value")) if rayleigh_match else None,
        "checksum": checksum_match.group("value") if checksum_match else None,
        "output": output,
    }


def run_shared_program(executable, node, n, nz, mode, threads=1,
                       scheduling=None, chunk=1024):
    command = [
        "srun", f"--nodelist={node}", str(executable),
        "-n", str(n), "-nz", str(nz), "-m", mode, "-s", "111"
    ]

    if executable.name != "seq":
        command += ["-t", str(threads)]

    if scheduling is not None:
        command += ["--scheduling", scheduling, "--chunk-size", str(chunk)]
    elif executable.name not in ("seq", "thread"):
        command += ["--chunk-size", str(chunk)]

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


def run_repetitions(executable, node, n, nz, mode, threads=1,
                    scheduling=None, chunk=1024, repetitions=3):
    results = []

    for repetition in range(1, repetitions + 1):
        result = run_shared_program(
            executable, node, n, nz, mode,
            threads, scheduling, chunk
        )
        result["repetition"] = repetition
        results.append(result)

    return results


def run_sequential(seq_executable, node, n, nz, mode):
    return run_shared_program(seq_executable, node, n, nz, mode)


def get_reference_rayleigh(seq_executable, node, n, nz, mode, cache):
    key = (n, nz, mode)

    if key not in cache:
        result = run_sequential(seq_executable, node, n, nz, mode)

        if result["returncode"] != 0 or result["rayleigh"] is None:
            print("Experiment failed: unable to obtain sequential Rayleigh reference")
            raise SystemExit(1)

        cache[key] = result["rayleigh"]

    return cache[key]


def check_rayleigh(rayleigh, reference_rayleigh, tolerance):
    if rayleigh is None:
        print("Experiment failed: Rayleigh value was not found")
        raise SystemExit(1)

    error = abs(rayleigh - reference_rayleigh)

    if error > tolerance:
        print(
            "Experiment failed due to insufficient result accuracy: "
            f"|rayleigh - reference| = {error:.12e} > "
            f"tolerance = {tolerance:.12e}"
        )
        raise SystemExit(1)

    return error


def write_csv(path, rows):
    if not rows:
        return

    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)


    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_result(rows, experiment, executable, implementation, n, nz, mode,
               threads, repetition, result, reference_rayleigh, tolerance=1e-9,
               scheduling="", chunk=0):
    error = check_rayleigh(
        result["rayleigh"],
        reference_rayleigh,
        tolerance
    )

    rows.append({
        "experiment": experiment,
        "executable": executable.name,
        "implementation": implementation,
        "n": n,
        "nz": nz,
        "mode": mode,
        "threads": threads,
        "scheduling": scheduling,
        "chunk_size": chunk,
        "repetition": repetition,
        "time_s": result["time"],
        "rayleigh": result["rayleigh"],
        "reference_rayleigh": reference_rayleigh,
        "rayleigh_abs_error": error,
        "checksum": result["checksum"],
        "returncode": result["returncode"],
    })
