# SPDX-License-Identifier: AGPL-3.0-or-later
"""Versioned external evidence adapters; raw provider outputs are always preserved."""
from __future__ import annotations
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from .runner import atomic_json, digest_file, run_process, utc_now
from .model import validate_record, is_comparable


def checked_executable(value):
    resolved = shutil.which(str(value))
    if not resolved: raise ValueError(f"executable unavailable: {value}")
    return str(Path(resolved).resolve())


def capture(command, output, *, timeout=300):
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("stdout.txt", "stderr.txt", "invocation.json")):
        raise ValueError("adapter output already contains evidence; choose a new directory")
    code, state = run_process(command, timeout=timeout, stdout=output / "stdout.txt", stderr=output / "stderr.txt")
    record = {"schema_version": 1, "argv": command, "returncode": code, "state": state,
              "recorded_at": utc_now(), "instrumented": True, "counts_as_core_benchmark_result": False}
    atomic_json(output / "invocation.json", record)
    return record


def instrument(kind, binary, item, output, *, tool="memcheck", kernel_regex=None, device=0, timeout=300):
    binary = checked_executable(binary)
    argv = [binary, *item.argv(), "--device", str(device), "--samples", "1", "--warmup", "0", "--max-repeats", "1", "--target-ms", "0.01"]
    if kind == "sanitizer":
        command = [checked_executable("compute-sanitizer"), "--tool", tool, "--error-exitcode", "99", *argv]
    else:
        command = [checked_executable("ncu"), "--set", "full", "--clock-control", "none", "--cache-control", "none",
                   "--target-processes", "all", "--launch-count", "8", "--force-overwrite",
                   "--export", str(output.resolve() / "profile")]
        if kernel_regex: command += ["--kernel-name", "regex:" + kernel_regex]
        command += argv
    result = capture(command, output, timeout=timeout)
    result["case"] = item.as_dict()
    worker_rows = []
    for line in (output / "stdout.txt").read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict) and value.get("schema_version") == 1: worker_rows.append(value)
        except json.JSONDecodeError: pass
    result["worker_outcomes"] = [{"status": row.get("status"), "validation": row.get("validation"), "reason": row.get("reason")} for row in worker_rows]
    result["benchmark_executed"] = bool(worker_rows) and all(not validate_record(row) and is_comparable(row) for row in worker_rows)
    result["verification_passed"] = kind == "sanitizer" and result["returncode"] == 0 and result["benchmark_executed"]
    result["warning"] = "instrumented timing is not used in normal performance comparisons; without --kernel-regex this is a trace of the first eight kernels and may capture initialization/autotuning instead of the final kernel"
    atomic_json(output / "invocation.json", result)
    return result


def disassemble(binary, output):
    binary = Path(binary).resolve()
    if not binary.is_file(): raise ValueError("binary missing")
    command = [checked_executable("cuobjdump"), "--dump-sass", "--dump-resource-usage", str(binary)]
    result = capture(command, output)
    result["binary_sha256"] = digest_file(binary)
    if result["returncode"] == 0:
        text = (output / "stdout.txt").read_text(errors="replace")
        # Inventory only: counts across the binary do not establish a specific case path.
        names = re.findall(r"/\*\s*[0-9a-fA-F]+\s*\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)", text)
        result["instruction_inventory"] = dict(Counter(name.split(".")[0] for name in names))
        result["interpretation"] = "static binary inventory; map kernel symbol to measured case before asserting native execution"
    atomic_json(output / "disassembly.json", result)
    return result


def cutlass_build(source, build_dir, architecture, kernels, jobs=4, nvcc=None):
    source, build_dir = Path(source).resolve(), Path(build_dir).resolve()
    if not (source / "tools/profiler/CMakeLists.txt").is_file(): raise ValueError("source is not a CUTLASS checkout")
    if architecture not in ("80", "89", "120a"): raise ValueError("invalid architecture")
    build_dir.mkdir(parents=True, exist_ok=True)
    git = checked_executable("git")
    revision = subprocess.run([git, "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15)
    dirty = subprocess.run([git, "-C", str(source), "status", "--porcelain"], capture_output=True, text=True, timeout=15)
    if revision.returncode: raise ValueError("CUTLASS checkout must have a recorded Git revision")
    cmake = checked_executable("cmake")
    command = [cmake, "-S", str(source), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
               "-DCUTLASS_NVCC_ARCHS=" + architecture, "-DCUTLASS_LIBRARY_KERNELS=" + kernels,
               "-DCUTLASS_UNITY_BUILD_ENABLED=ON", "-DCUTLASS_ENABLE_TESTS=OFF", "-DCUTLASS_ENABLE_EXAMPLES=OFF"]
    if nvcc: command += ["-DCMAKE_CUDA_COMPILER=" + str(Path(nvcc).resolve())]
    result = capture(command, build_dir / "configure", timeout=1800)
    if result["returncode"] == 0:
        result = capture([cmake, "--build", str(build_dir), "--target", "cutlass_profiler", "--parallel", str(jobs)], build_dir / "compile", timeout=86400)
    result.update({"source_revision": revision.stdout.strip(), "dirty_source": bool(dirty.stdout),
                   "source_status": dirty.stdout, "architecture": architecture, "kernel_filter": kernels})
    atomic_json(build_dir / "archbench_cutlass_build.json", result)
    return result


def parse_cutlass_csv(path):
    """Keep every row. A successful launch alone is not a verified result."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            normalized = {str(k).strip().lower(): str(v).strip() for k, v in row.items() if k is not None}
            status = normalized.get("status", "unknown")
            disposition = normalized.get("disposition", normalized.get("verification", "unknown"))
            runtime = normalized.get("runtime", normalized.get("runtime (ms)", ""))
            try: runtime = float(runtime)
            except ValueError: runtime = None
            if runtime is not None and not math.isfinite(runtime): runtime = None
            valid = status.lower() == "success" and disposition.lower() == "passed" and runtime is not None and runtime > 0
            rows.append({"provider": "cutlass", "raw": row, "validated_measured": valid,
                         "status": status, "disposition": disposition, "runtime_ms": runtime,
                         "timing_scope": "CUTLASS reported average; not an archbench sample distribution"})
    return rows


def cutlass_run(profiler, output, arguments, *, timeout=3600, enumerate_only=False):
    profiler = checked_executable(profiler)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("profiler*.csv")): raise ValueError("output contains old profiler CSV files; choose a new directory")
    # Control verification and destinations; user arguments may select shapes/kernels/math.
    forbidden = ("--verification-enabled", "--output", "--append", "--mode", "--profiling-enabled")
    if any(arg.split("=", 1)[0] in forbidden for arg in arguments):
        raise ValueError("adapter controls mode, output and verification; remove conflicting arguments")
    command = [profiler, *arguments]
    if enumerate_only:
        command += ["--mode=enumerate"]
    else:
        command += ["--verification-enabled=true", "--profiling-enabled=true", "--report-not-run=true", "--append=false",
                    "--mode=profile", "--output=" + str(output / "profiler")]
    result = capture(command, output, timeout=timeout)
    result["profiler_sha256"] = digest_file(Path(profiler))
    all_rows = []
    for path in sorted(output.glob("profiler*.csv")):
        parsed = parse_cutlass_csv(path)
        for row in parsed: row["source_csv"] = path.name
        all_rows.extend(parsed)
    with (output / "provider_results.jsonl").open("w") as stream:
        for row in all_rows: stream.write(json.dumps(row, allow_nan=False) + "\n")
    result["rows"] = len(all_rows)
    result["validated_measured"] = sum(row["validated_measured"] for row in all_rows)
    result["not_run_or_failed_or_unverified"] = len(all_rows) - result["validated_measured"]
    if not enumerate_only and result["returncode"] == 0 and not all_rows:
        result["warning"] = "no machine-readable kernel results; zero coverage, not success"
    atomic_json(output / "summary.json", result)
    return result


def external_run(spec_path, output, timeout=3600):
    spec = json.loads(Path(spec_path).read_text())
    command = spec.get("argv")
    if not isinstance(command, list) or not command or any(not isinstance(x, str) for x in command):
        raise ValueError("external specification requires an argv array, never a shell string")
    command[0] = checked_executable(command[0])
    result = capture(command, Path(output), timeout=timeout)
    result["provider"] = spec.get("provider", "external")
    result["validation"] = "not_inferred: inspect provider's numerical verification output"
    result["counts_as_core_benchmark_result"] = False
    atomic_json(Path(output) / "invocation.json", result)
    return result
