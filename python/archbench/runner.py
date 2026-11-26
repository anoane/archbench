# SPDX-License-Identifier: AGPL-3.0-or-later
"""Isolated workers, resume guards, immutable metadata, and complete status accounting."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import time
import uuid
from typing import Iterable
from .model import Case, SCHEMA_VERSION, canonical, validate_record, is_measured_pass
from .telemetry import NvmlSampler
from .pairing import pairing_index, pairing_metadata, pair_summary

RUN_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
                "CUBLAS_EMULATION_STRATEGY", "CUBLAS_EMULATE_SINGLE_PRECISION", "CUBLAS_EMULATE_DOUBLE_PRECISION",
                "CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK", "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT",
                "CUDA_FORCE_PTX_JIT", "CUDA_DISABLE_PTX_JIT", "CUDA_MODULE_LOADING", "LD_LIBRARY_PATH", "LD_PRELOAD")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): hasher.update(block)
    return hasher.hexdigest()


def selection_source(args) -> dict:
    """Describe the effective input without changing the case-based resume identity.

    The legacy top-level ``profile`` records the CLI argument even when a matrix
    takes precedence. This additive field identifies the source actually used.
    It is written for new runs only; old run provenance is never reconstructed.
    """
    matrix = getattr(args, "matrix", None)
    if matrix is not None:
        path = Path(matrix).expanduser().resolve()
        source = {"kind": "jsonl_plan" if path.suffix == ".jsonl" else "json_matrix",
                  "path": str(path), "sha256": digest_file(path),
                  "profile_argument_ignored": True}
    else:
        source = {"kind": "profile", "profile": args.profile}
    source["filters"] = {name: value for name, default in
                         (("families", ""), ("dtypes", ""), ("ops", ""),
                          ("match", ""), ("shard", None), ("limit", 0))
                         if (value := getattr(args, name, default)) != default}
    explicit = getattr(args, "software_pairs", None)
    source["software_pairs"] = (not (matrix and Path(matrix).suffix == ".jsonl")) if explicit is None else explicit
    source["shard_basis"] = "operation_id" if source["software_pairs"] else "case_id"
    return source


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip(): continue
            try: yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON (including a truncated final line): {error}") from error


def repair_tail(path: Path) -> bool:
    """Only remove an incomplete final write. Corrupt completed records remain a hard error."""
    if not path.exists() or path.stat().st_size == 0: return False
    with path.open("rb+") as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n": return False
        size = stream.tell()
        cursor = size
        while cursor:
            amount = min(cursor, 65536)
            cursor -= amount
            stream.seek(cursor)
            block = stream.read(amount)
            last = block.rfind(b"\n")
            if last >= 0:
                stream.truncate(cursor + last + 1)
                return True
        stream.truncate(0)
    return True


def run_process(argv: list[str], *, timeout: float, stdout: Path, stderr: Path, env=None) -> tuple[int | None, str]:
    """No shell. Kill the complete worker process group on timeout/interruption."""
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(argv, stdout=out, stderr=err, env=env, start_new_session=True)
        try:
            return process.wait(timeout=timeout), "completed"
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()
            return None, "timeout"
        except KeyboardInterrupt:
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.wait()
            return None, "interrupted"


def synthetic_record(status: str, reason: str) -> dict:
    return {"schema_version": SCHEMA_VERSION, "status": status, "reason": reason, "validation": "not_run",
            "samples_ms": [], "execution": "not_executed", "metadata": {}, "metrics": {}}


def invoke_worker(binary: Path, argv: list[str], out_dir: Path, stem: str, timeout: float):
    stdout = out_dir / f"{stem}.stdout.jsonl"
    stderr = out_dir / f"{stem}.stderr.txt"
    start = time.monotonic_ns()
    try:
        code, state = run_process([str(binary), *argv], timeout=timeout, stdout=stdout, stderr=stderr)
    except OSError as error:
        return synthetic_record("error", f"worker could not start: {error}")
    if state != "completed": record = synthetic_record(state, f"worker {state}; limit {timeout:g} seconds")
    else:
        try:
            rows = list(read_jsonl(stdout))
            if len(rows) != 1 or not isinstance(rows[0], dict): raise ValueError("expected exactly one JSON object")
            record = rows[0]
            if record.get("execution") != "metadata_only":
                errors = validate_record(record)
                if errors: raise ValueError("; ".join(errors))
            if code != 0 and record.get("status") == "ok":
                raise ValueError(f"worker exited {code} despite ok status")
        except (ValueError, OSError, TypeError) as error:
            record = synthetic_record("error", f"invalid worker result: {error}; inspect raw logs")
    record["process"] = {"returncode": code, "start_monotonic_ns": start,
                         "end_monotonic_ns": time.monotonic_ns(),
                         "stdout": str(stdout.name), "stderr": str(stderr.name)}
    return record



def validated_results(path: Path, expected_cases: dict[str, Case], run_id: str):
    for row in read_jsonl(path):
        if not isinstance(row, dict): raise ValueError("result row must be an object")
        item = Case.from_dict(row.get("case", {}))
        if item.id not in expected_cases: raise ValueError("result contains a case outside its plan")
        if item.as_dict() != expected_cases[item.id].as_dict(): raise ValueError("result case differs from planned case")
        if row.get("run_id") != run_id: raise ValueError("result belongs to a different run")
        errors = validate_record(row)
        if errors: raise ValueError("invalid saved result: " + "; ".join(errors))
        yield row


def read_plan(directory: Path, metadata):
    plan = [Case.from_dict(item) for item in read_jsonl(directory / "plan.jsonl")]
    expected = {item.id: item for item in plan}
    if len(expected) != len(plan) or len(plan) != metadata["selected_cases"]:
        raise ValueError("saved plan case count or uniqueness is inconsistent")
    digest = hashlib.sha256("\n".join(item.id for item in plan).encode()).hexdigest()
    if digest != metadata["identity"]["selection_sha256"]: raise ValueError("saved plan fingerprint mismatch")
    return expected


def read_run(directory: Path):
    metadata = json.loads((directory / "run.json").read_text())
    expected = read_plan(directory, metadata)
    return metadata, {row["case"]["id"]: row for row in validated_results(directory / "results.jsonl", expected, metadata["run_id"])}


def resolve_gpu(binary, args, logs):
    """Match a full UUID against CUDA-visible ordinals, not nvidia-smi indices."""
    requested = getattr(args, "gpu_uuid", None)
    wanted = None
    if requested:
        wanted = requested.removeprefix("GPU-").replace("-", "").lower()
        if len(wanted) != 32 or any(c not in "0123456789abcdef" for c in wanted):
            raise ValueError("--gpu-uuid requires a full 16-byte GPU UUID (GPU-... or 32 hexadecimal digits)")
    first = args.device if wanted is None else 0
    info = invoke_worker(binary, ["--info", "--device", str(first)], logs, "device", min(args.timeout, 60))
    if info.get("status") != "ok": raise ValueError(f"cannot initialize GPU: {info.get('reason')}")
    if wanted is None: return first, info
    metadata = info.get("metadata", {})
    count = int(metadata.get("visible_device_count", 1))
    if not 1 <= count <= 1024: raise ValueError("worker reported invalid visible_device_count")
    for ordinal in range(count):
        current = info if ordinal == 0 else invoke_worker(binary, ["--info", "--device", str(ordinal)], logs,
                    f"device_candidate_{ordinal}", min(args.timeout, 60))
        if current.get("status") == "ok" and current.get("metadata", {}).get("uuid_hex", "").lower() == wanted:
            return ordinal, current
    raise ValueError("requested GPU UUID is not visible to CUDA; check CUDA_VISIBLE_DEVICES and the full UUID")

def run_suite(cases: Iterable[Case], args) -> dict:
    # Linux/WSL2 scope: protect immutable run provenance and append-only evidence.
    import fcntl
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise ValueError("another process owns this run directory") from error
        try:
            result = _run_suite_locked(cases, args)
            if getattr(args, "report", True):
                from .report import render_report
                rendered = render_report(output)
                result["report_html"] = str(rendered)
                result["report_csv"] = str(output / "results.csv")
                result["pair_coverage_json"] = str(output / "pairs.json")
                atomic_json(output / "summary.json", result)
            return result
        finally: fcntl.flock(lock, fcntl.LOCK_UN)


def _run_suite_locked(cases: Iterable[Case], args) -> dict:
    binary = Path(args.binary).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not binary.is_file(): raise ValueError(f"CUDA worker is missing: {binary}; run ./bench build first")
    if not os.access(binary, os.X_OK): raise ValueError(f"worker is not executable: {binary}")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    result_path, metadata_path = output / "results.jsonl", output / "run.json"
    if (result_path.exists() or metadata_path.exists()) and not args.resume:
        raise ValueError("output already contains a run; use --resume or a new output directory")
    cases = list(cases)
    if not cases: raise ValueError("no cases selected")
    if any(item.provider != "core" for item in cases): raise ValueError("run accepts core provider cases only; use the explicit provider adapter")
    expected_cases = {item.id: item for item in cases}
    if len(expected_cases) != len(cases): raise ValueError("duplicate case IDs in selected plan")
    selection_digest = hashlib.sha256("\n".join(x.id for x in cases).encode()).hexdigest()
    args.device, info = resolve_gpu(binary, args, logs)
    resolved_max_samples = getattr(args, "max_samples", None)
    if resolved_max_samples is None: resolved_max_samples = max(21, args.samples)
    policy = {"samples": args.samples, "warmup": args.warmup, "target-ms": args.target_ms,
              "max-samples": resolved_max_samples, "target-cv": getattr(args, "target_cv", .05),
              "warmup-ms": getattr(args, "warmup_ms", 0), "repeats": getattr(args, "repeats", 0),
              "max-repeats": args.max_repeats, "memory-fraction": args.memory_fraction,
              "memory-limit-mib": args.memory_limit_mib, "device": args.device}
    env = {key: os.environ.get(key) for key in RUN_ENV_KEYS}
    identity = {"binary_sha256": digest_file(binary), "policy": policy, "environment": env,
                "selection_sha256": selection_digest, "timeout_seconds": args.timeout, "shuffle_seed": args.shuffle_seed}
    device_metadata = info.get("metadata")
    if not isinstance(device_metadata, dict): raise ValueError("worker --info metadata must be an object")
    gpu_uuid = device_metadata.get("uuid_hex")
    if not isinstance(gpu_uuid, str) or len(gpu_uuid) != 32 or any(c not in "0123456789abcdefABCDEF" for c in gpu_uuid):
        raise ValueError("worker --info did not report a valid 16-byte GPU UUID")
    identity["gpu_uuid"] = gpu_uuid
    identity["software"] = {key: device_metadata.get(key) for key in ("source_sha256", "build_flags_sha256",
        "nvcc_version", "runtime_version", "driver_api_version", "compile_cudart_version", "cublas_version", "cublaslt_version", "cufft_version", "cusparse_version")}
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            driver_query = subprocess.run([smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=10)
            identity["software"]["driver_version"] = sorted(set(driver_query.stdout.strip().splitlines())) if driver_query.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            identity["software"]["driver_version"] = None
    else: identity["software"]["driver_version"] = None
    if args.resume:
        if not metadata_path.exists(): raise ValueError("resume requires run.json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("identity") != identity:
            raise ValueError("resume identity differs (binary, GPU, case selection, environment or measurement policy); use a new directory")
        if repair_tail(result_path): print("Discarded an incomplete final result line from interrupted write.", file=sys.stderr)
        read_plan(output, metadata)
    else:
        metadata = {"schema_version": SCHEMA_VERSION, "run_id": str(uuid.uuid4()), "created_at": utc_now(),
                    "identity": identity, "binary": str(binary), "selected_cases": len(cases),
                    "device": info.get("metadata"), "host": {"system": platform.system(), "release": platform.release(),
                    "machine": platform.machine(), "python": platform.python_version()},
                    "profile": args.profile, "shuffle_seed": args.shuffle_seed,
                    "selection_source": selection_source(args),
                    "measurement_preset": getattr(args, "quality", "custom"),
                    "requested_gpu_uuid": getattr(args, "gpu_uuid", None),
                    "timing_warning": "Event batches include enqueue gaps; samples are batch averages, not individual-call latency tails.",
                    "telemetry_scope": "whole worker including CUDA initialization, setup, validation and timing"}
        atomic_json(metadata_path, metadata)
        with (output / "plan.jsonl").open("w", encoding="utf-8") as stream:
            for item in cases: stream.write(canonical(item.as_dict()) + "\n")
    history = list(validated_results(result_path, expected_cases, metadata["run_id"])) if result_path.exists() else []
    previous = {row["case"]["id"]: row for row in history}
    if args.retry_failed:
        done = {key for key, row in previous.items() if is_measured_pass(row)}
    else: done = set(previous)
    random.Random(args.shuffle_seed).shuffle(cases)
    sampler = NvmlSampler(output / "telemetry.jsonl", identity["gpu_uuid"], args.telemetry_interval) if args.telemetry else None
    if sampler:
        sampler.start()
        if sampler.error: print(f"Telemetry unavailable: {sampler.error}", file=sys.stderr)
        metadata["telemetry_error"] = sampler.error
        atomic_json(metadata_path, metadata)
    counters = Counter()
    common = [value for key, val in policy.items() for value in ("--" + key, str(val))]
    stopped = False
    attempts = Counter(row["case"]["id"] for row in history)
    pairs = pairing_index(cases)
    try:
        with result_path.open("a", encoding="utf-8") as stream:
            remaining = [item for item in cases if item.id not in done]
            for index, item in enumerate(remaining, 1):
                attempts[item.id] += 1
                stem = f"{item.id}.{attempts[item.id]}"
                if sampler: sampler.case_id = item.id
                record = invoke_worker(binary, item.argv() + common, logs, stem, args.timeout)
                record.update({"case": item.as_dict(), "run_id": metadata["run_id"], "recorded_at": utc_now(),
                               "attempt": attempts[item.id], "pairing": pairing_metadata(item, pairs)})
                stream.write(canonical(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                counters[record["status"]] += 1
                if index == 1 or index % args.progress_every == 0 or record["status"] not in ("ok", "unsupported"):
                    print(f"[{index}/{len(remaining)}] {item.id}: {record['status']} {record.get('reason', '')[:180]}", file=sys.stderr, flush=True)
                if record["status"] == "interrupted" or (args.fail_fast and record["status"] in ("error", "validation_failed")):
                    stopped = True
                    break
    finally:
        if sampler: sampler.stop()
    latest = {row["case"]["id"]: row for row in validated_results(result_path, expected_cases, metadata["run_id"])}
    summary = {"run_id": metadata["run_id"], "selected": len(cases), "recorded": len(latest),
               "missing": len(cases) - len(latest), "validated_measured": sum(is_measured_pass(row) for row in latest.values()),
               "status_counts": dict(Counter(row["status"] for row in latest.values())), "stopped": stopped,
               "completed_at": utc_now()}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "pairs.json", pair_summary(cases, latest))
    return summary
