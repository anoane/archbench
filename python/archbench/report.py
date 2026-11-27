# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-contained HTML and exact-case cross-device comparison. No synthetic scores."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from .model import VALIDATED, is_measured_pass, is_comparable
from .runner import atomic_json, read_run, read_plan
from .pairing import DEFAULT_BACKENDS, pair_summary, pairing_index, pairing_metadata
from .statistics import assess_quality, describe, rate_summary


def latest_results(directory: Path):
    return read_run(directory)[1]


def summary(directory: Path):
    run, rows = read_run(directory)
    return _summarize(run, rows)


def _summarize(run, rows):
    by_family = defaultdict(Counter)
    for row in rows.values():
        family = row["case"]["family"]
        by_family[family][row["status"]] += 1
        if is_measured_pass(row): by_family[family]["validated_measured"] += 1
    return {"run": run, "recorded": len(rows), "selected": run["selected_cases"],
            "missing": run["selected_cases"] - len(rows),
            "validated_measured": sum(is_measured_pass(row) for row in rows.values()),
            "statuses": dict(Counter(row["status"] for row in rows.values())),
            "families": dict(by_family)}


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _format_number(value):
    value = _number(value)
    return format(value, ".6g") if value is not None else "N/A"


def _format_time(milliseconds):
    value = _number(milliseconds)
    if value is None or value <= 0: return "N/A"
    return f"{_format_number(value * 1000)} µs"


def _visible_execution(record):
    """Correct an old default label for a skipped benchmark; keep raw evidence."""
    raw = record.get("execution")
    skipped = {"unsupported", "not_implemented", "missing_dependency", "missing_binary",
               "architecture_mismatch", "compile_failed", "probe_only", "memory_budget"}
    if (record.get("status") in skipped and not record.get("samples_ms") and
            raw in (None, "compiler_lowered", "not_executed")):
        return "not_executed"
    return raw or "unknown"


def _work_domain(record):
    """A work counter only becomes FLOPs when its numerical semantics justify it."""
    item = record["case"]
    family, params = item["family"], item["params"]
    metadata, unit = record.get("metadata", {}), record.get("unit", "")
    explicit = metadata.get("work_domain") or record.get("work_domain")
    if family == "blas" and params.get("op") in ("asum", "nrm2", "iamax", "iamin"):
        return "reduction_work", "operation_scope_overrides_generic_float_label"
    if unit == "Gbit-pair/s": return "bit_pair", "explicit_unit"
    if explicit: return explicit, "explicit_metadata"
    if unit.lower() in ("gflop/s", "gflops", "tflop/s"): return "floating_point", "explicit_unit"
    unit_domains = {"Gatomic/s": "atomic", "Gload/s": "load", "Glaunch/s": "launch",
                    "Gbarrier/s": "barrier", "Gitem/s": "item", "Gvalue/s": "value",
                    "Gconversion/s": "conversion", "Gcall/s": "special_function"}
    if unit in unit_domains: return unit_domains[unit], "explicit_unit"
    dtype = params.get("dtype", "f32")
    integer = dtype.startswith(("i", "u")) and dtype[1:].isdigit()
    if family in ("integer", "integer_width"): return "integer", "family_and_type"
    if family == "bit": return "bit", "family_and_type"
    if family == "sfu": return "special_function", "family_and_type"
    if family == "convert": return "conversion", "family_and_type"
    if family in ("scalar", "packed", "tensor", "gemm", "blas", "quant_gemm", "sparse"):
        if dtype.startswith("b1"): return "bit_pair", "family_and_type"
        return ("integer" if integer else "floating_point"), "family_and_type"
    if family == "quant": return "value", "family_and_type"
    if family == "fft": return "item", "family_and_type"
    if family == "workload": return ("floating_point" if params.get("op") == "attention" else "item"), "family_and_type"
    if family == "memory":
        if params.get("op") in ("read", "shared_bank", "pointer_chase"): return "load", "family_and_type"
        if params.get("op") == "triad": return "floating_point", "family_and_type"
        return "data_movement", "family_and_type"
    return "generic_work", "unspecified"


WORK_UNITS = {
    "floating_point": ("GFLOP/s", "TFLOP/s"), "integer": ("GOP/s (integer)", "TOP/s (integer)"),
    "bit": ("Gop/s (bitwise)", "Top/s (bitwise)"), "bit_pair": ("Gbit-pair/s", "Tbit-pair/s"),
    "special_function": ("Gcall/s", "Tcall/s"), "conversion": ("Gconversion/s", "Tconversion/s"),
    "atomic": ("Gatomic/s", "Tatomic/s"), "launch": ("Glaunch/s", "Tlaunch/s"),
    "barrier": ("Gbarrier/s", "Tbarrier/s"), "load": ("Gload/s", "Tload/s"),
    "value": ("Gvalue/s", "Tvalue/s"), "item": ("Gitem/s", "Titem/s"),
    "reduction_work": ("Gop/s (reduction model)", "Top/s (reduction model)"),
    "generic_work": ("Gwork/s", "Twork/s"), "data_movement": ("Gwork/s", "Twork/s")}

SINGLE_REPORT_LIMIT = 5000
REPORT_SHARD_ROWS = 1000


def _classification(record):
    status, execution = record.get("status"), _visible_execution(record)
    metadata = record.get("metadata", {})
    unavailable = {"unsupported": "backend_request", "not_implemented": "unimplemented",
                   "missing_dependency": "installation", "missing_binary": "installation",
                   "architecture_mismatch": "architecture", "memory_budget": "resources",
                   "out_of_memory": "resources", "compile_failed": "build", "probe_only": "compile_only"}
    availability = "unavailable" if status in unavailable else (
        "available" if status in ("ok", "measurement_only", "validation_failed") or record.get("samples_ms") else "unknown")
    capability = {"availability": availability,
                  "scope": metadata.get("support_scope", unavailable.get(status, "tested_request" if availability == "available" else "unknown")),
                  "reason": record.get("reason", ""),
                  "claim": "this backend/request/environment only; unavailable does not mean globally impossible"}
    codegen = metadata.get("tensor_codegen_evidence", metadata.get("codegen_evidence", "not_inspected"))
    if execution == "not_executed": kind, label = "not_executed", "Not executed"
    elif execution == "compiler_expanded_tensor_ptx" or codegen in ("compiler_expanded_mma", "scalar_expansion"):
        kind, label = "compiler_emulation", "Compiler-expanded arithmetic; may use wider hardware instructions"
    elif execution == "promoted_ptx_with_width_wrap":
        kind, label = "promoted_arithmetic", "Wider PTX arithmetic with explicit requested-width wrapping"
    elif metadata.get("fixedpoint_used_last_call") == "true":
        kind, label = "library_software_emulation", "Library emulation reported for last call"
    elif "emulat" in execution or "software" in execution:
        kind, label = "software", "Software arithmetic / reference implementation"
    elif codegen == "direct_mma_sass":
        kind, label = "native_build_instruction", "Direct MMA in compiler-probe SASS; runtime correctness is separate"
    elif execution.startswith("requested_") or execution == "native":
        kind, label = "native_requested", "Native instruction/API requested; SASS not established"
    elif "library" in execution:
        kind, label = "library_selected_unknown", "Library selected; hardware path unverified"
    elif execution == "runtime_api": kind, label = "runtime", "CUDA runtime operation"
    else: kind, label = "compiler_selected_unknown", "Compiler/intrinsic lowering; hardware path unverified"
    return capability, {"class": kind, "label": label, "execution": execution,
                        "codegen_evidence": codegen,
                        "evidence_scope": metadata.get("tensor_codegen_evidence_scope", "built instruction selection; this alone does not establish successful runtime execution"),
                        "evidence": metadata.get("execution_evidence", metadata.get("sass_evidence", codegen))}


def _display_row(record, policy=None):
    """Derive presentation fields without changing the result or its metrics."""
    item = record["case"]
    stats = describe(record.get("samples_ms", []))
    metrics = dict(record.get("metrics", {}))
    metadata = dict(record.get("metadata", {}))
    work, byte_count = record.get("work_per_iteration"), record.get("bytes_per_iteration")
    unit = record.get("unit")
    domain, domain_source = _work_domain(record)
    useful_unit, tera_unit = WORK_UNITS.get(domain, WORK_UNITS["generic_work"])
    samples = record.get("samples_ms", [])
    giga = rate_summary(samples, work, unit=useful_unit, timing_stats=stats)
    tera = rate_summary(samples, work, scale=1e12, unit=tera_unit, timing_stats=stats)
    bandwidth = rate_summary(samples, byte_count, unit="GB/s", timing_stats=stats)
    binary_bandwidth = rate_summary(samples, byte_count, scale=2**30, unit="GiB/s", timing_stats=stats)
    useful_rate, byte_rate = (giga["value"] if giga else None), (bandwidth["value"] if bandwidth else None)
    policy = policy or {}
    threshold_source = "run_policy" if "target-cv" in policy else "worker_policy" if "timing_target_cv" in metrics else "post_hoc_legacy_report_default"
    target_cv = policy.get("target-cv", metrics.get("timing_target_cv", .05))
    if _number(target_cv) is None or not 0 <= target_cv <= 1:
        target_cv, threshold_source = .05, "post_hoc_default_invalid_recorded_threshold"
    quality = assess_quality(stats, target_cv=target_cv)
    quality["threshold_source"] = threshold_source
    if threshold_source.startswith("post_hoc"):
        quality["warnings"].append("CV threshold is a post-hoc report classification; no adaptive collection policy is inferred for this record")
    capability, implementation = _classification(record)
    checked = record.get("validation") in VALIDATED | {"failed"}
    absolute = _number(record.get("max_abs_error")) if checked else None
    relative = _number(record.get("max_rel_error")) if checked else None
    error_note = ""
    if item["family"] == "quant" and not metadata.get("relative_error_definition"):
        relative = None
        if checked:
            error_note = "Relative error unavailable: this legacy quant record did not compute it."
    notes = [record["reason"]] if record.get("reason") else []
    if checked:
        notes.append(f"max abs error: {_format_number(absolute)}; max rel error: {_format_number(relative)}")
    else:
        notes.append("Numerical error: N/A (validation not run or not applicable).")
    if error_note: notes.append(error_note)
    shown_rate = tera if domain in ("floating_point", "integer") else giga
    fft_estimate = rate_summary(samples, metrics.get("estimated_fft_flops"), scale=1e12,
                                unit="estimated TFLOP/s (FFT model)", timing_stats=stats)
    return {"id": item["id"], "family": item["family"], "params": item["params"],
            "status": record["status"], "validation": record.get("validation"),
            "execution": _visible_execution(record), "recorded_execution": record.get("execution"),
            "unit": unit, "useful_rate_unit": useful_unit, "useful_tera_rate_unit": tera_unit,
            "useful_tera_rate": tera["value"] if tera else None,
            "work_domain": domain, "work_domain_source": domain_source,
            "capability": capability, "implementation": implementation, "quality": quality,
            "pairing": record.get("pairing"),
            "rates": {"useful_giga": giga, "useful_tera": tera, "logical_gbytes": bandwidth,
                      "logical_gibytes": binary_bandwidth, "estimated_fft_tflops": fft_estimate},
            "tflops": tera["value"] if tera and domain == "floating_point" else None,
            "integer_tops": tera["value"] if tera and domain == "integer" else None,
            "estimated_fft_tflops": fft_estimate["value"] if fft_estimate else None,
            "logical_gibytes_s": binary_bandwidth["value"] if binary_bandwidth else None,
            "useful_work_per_ns": useful_rate,
            "ns_per_useful_work": stats["median_ns"] / work if giga else None,
            "useful_work_per_logical_byte": work / byte_count if giga and bandwidth else None,
            "work_per_iteration": work, "bytes_per_iteration": byte_count,
            "stats": stats, "metrics": metrics, "metadata": metadata,
            "useful_gops": useful_rate, "logical_gbytes_s": byte_rate,
            "max_abs_error": absolute, "max_rel_error": relative, "error": absolute,
            "abs_error_available": absolute is not None, "rel_error_available": relative is not None,
            "recorded_max_abs_error": record.get("max_abs_error"),
            "recorded_max_rel_error": record.get("max_rel_error"), "error_note": error_note,
            "reason": record.get("reason"), "notes": " ".join(notes),
            "samples_ms": samples, "samples_us": [value * 1000 for value in samples],
            "timing_phases": {key: value for key, value in metrics.items() if key.endswith(("_host_ms", "_device_ms", "_setup_ms", "_preparation_ms"))},
            "display": {"median_time": _format_time(stats.get("median_ms")),
                        "useful_rate": f"{_format_number(shown_rate['value'])} {shown_rate['unit']}" if shown_rate else "N/A",
                        "logical_rate": f"{_format_number(byte_rate)} GB/s" if byte_rate is not None else "N/A"}}


def render_report(directory: Path, destination: Path | None = None):
    directory = directory.resolve()
    destination = destination or directory / "report.html"
    run, records = read_run(directory)
    overview = _summarize(run, records)
    planned = read_plan(directory, run)
    if len(planned) > SINGLE_REPORT_LIMIT:
        return _render_sharded_report(directory, destination, run, records, planned, overview)
    rows = [_display_row(records.get(case.id, {"case": case.as_dict(), "status": "missing",
        "validation": "not_run", "execution": "not_executed", "samples_ms": [],
        "reason": "Planned case has no saved result; availability and performance are not established.",
        "metadata": {"report_only": "derived placeholder for an unrecorded planned case"}}),
        run.get("identity", {}).get("policy", {})) for case in planned.values()]
    pair_index = pairing_index(planned.values())
    pairs = pair_summary(planned.values(), records)
    for row in rows:
        if row["pairing"] is None:
            row["pairing"] = {**pairing_metadata(planned[row["id"]], pair_index),
                              "provenance": "derived from the saved plan; original record unchanged"}
    display_by_id = {row["id"]: row for row in rows}
    for pair in pairs["rows"]:
        requested, software = records.get(pair["requested_case_id"]), records.get(pair["software_case_id"])
        pair["software_speedup_vs_requested"] = comparison_basis(requested, software)
        for role in ("requested", "software"):
            row = display_by_id.get(pair[role + "_case_id"])
            pair[role + "_measurement"] = ({"median_us": row["stats"].get("median_us"),
                "useful_rate": row["display"]["useful_rate"], "logical_bandwidth": row["display"]["logical_rate"],
                "implementation": row["implementation"], "quality": row["quality"], "reason": row["reason"]} if row else None)
        pair["interpretation"] = ("numerical reference; library policy equivalence is not claimed" if pair["relation"] == "mathematical_reference" else
                                   "same requested numeric controls; execution strategies can differ")
    overview["pairing_counts"] = pairs["counts"]
    overview["pairing_unique_operation_counts"] = pairs.get("unique_operation_counts", {})
    overview["measurement_quality_counts"] = dict(Counter(row["quality"]["label"] for row in rows))
    overview["implementation_counts"] = dict(Counter(row["implementation"]["class"] for row in rows))
    payload = json.dumps({"overview": overview, "rows": rows, "pairs": pairs}, allow_nan=False).replace("<", "\\u003c")
    destination.write_text(REPORT_TEMPLATE.replace("PAYLOAD", payload), encoding="utf-8")
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = _csv_writer(stream)
        for row in rows: _write_csv_row(writer, row)
    atomic_json(directory / "coverage.json", overview)
    atomic_json(directory / "pairs.json", pairs)
    return destination


def _partition_cases(planned, maximum=REPORT_SHARD_ROWS):
    """Hash prefixes permit exact-ID routing without an index entry per case."""
    families = defaultdict(list)
    for identifier, case in planned.items(): families[case.family].append(identifier)
    widths, groups = {}, {}
    for family, identifiers in sorted(families.items()):
        width = 1
        while True:
            buckets = defaultdict(list)
            for identifier in identifiers:
                buckets[identifier.split(".", 1)[1][:width]].append(identifier)
            if max(map(len, buckets.values())) <= maximum: break
            width += 1
        widths[family] = width
        for prefix, members in sorted(buckets.items()):
            groups[f"{family}.{prefix}.html"] = members
    return widths, groups


def _missing_record(case):
    return {"case": case.as_dict(), "status": "missing", "validation": "not_run",
            "execution": "not_executed", "samples_ms": [],
            "reason": "Planned case has no saved result; availability and performance are not established.",
            "metadata": {"report_only": "derived placeholder for an unrecorded planned case"}}


def _enrich_pair(pair, records, display_for):
    pair = dict(pair)
    requested, software = records.get(pair["requested_case_id"]), records.get(pair["software_case_id"])
    pair["software_speedup_vs_requested"] = comparison_basis(requested, software)
    for role in ("requested", "software"):
        row = display_for(pair[role + "_case_id"])
        pair[role + "_measurement"] = ({"median_us": row["stats"].get("median_us"),
            "useful_rate": row["display"]["useful_rate"], "logical_bandwidth": row["display"]["logical_rate"],
            "implementation": row["implementation"], "quality": row["quality"], "reason": row["reason"]} if row else None)
    pair["interpretation"] = ("numerical reference; library policy equivalence is not claimed" if pair["relation"] == "mathematical_reference" else
                               "same requested numeric controls; execution strategies can differ")
    return pair


def _render_sharded_report(directory, destination, run, records, planned, overview):
    """Bound derived-row memory and initial browser payload for very large plans."""
    widths, groups = _partition_cases(planned)
    detail_directory = destination.parent / (destination.stem + "_details")
    detail_directory.mkdir(parents=True, exist_ok=True)
    policy = run.get("identity", {}).get("policy", {})
    pair_index = pairing_index(case for case in planned.values() if case.family in DEFAULT_BACKENDS)
    pairs = pair_summary(planned.values(), records)
    pairs_by_id = defaultdict(list)
    for pair in pairs["rows"]:
        pairs_by_id[pair["requested_case_id"]].append(pair)
        pairs_by_id[pair["software_case_id"]].append(pair)
    quality_counts, implementation_counts = Counter(), Counter()
    overview["pairing_counts"] = pairs["counts"]
    overview["pairing_unique_operation_counts"] = pairs.get("unique_operation_counts", {})
    pages = []
    filenames = list(groups)
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = _csv_writer(stream)
        for position, (filename, identifiers) in enumerate(groups.items()):
            display_cache = {}
            def display_for(identifier):
                if identifier not in planned: return None
                if identifier not in display_cache:
                    case = planned[identifier]
                    row = _display_row(records[identifier] if identifier in records else _missing_record(case), policy)
                    if row["pairing"] is None and case.family in DEFAULT_BACKENDS:
                        row["pairing"] = {**pairing_metadata(case, pair_index),
                                          "provenance": "derived from saved plan; original record unchanged"}
                    display_cache[identifier] = row
                return display_cache[identifier]
            rows = [display_for(identifier) for identifier in identifiers]
            selected_pairs = {}
            for identifier in identifiers:
                for pair in pairs_by_id.get(identifier, []):
                    selected_pairs[pair["requested_case_id"]] = pair
            local_pairs = {**pairs, "rows": [_enrich_pair(pair, records, display_for) for pair in selected_pairs.values()]}
            for row in rows:
                _write_csv_row(writer, row)
                quality_counts[row["quality"]["label"]] += 1
                implementation_counts[row["implementation"]["class"]] += 1
            page = {"file": f"{detail_directory.name}/{filename}", "family": planned[identifiers[0]].family,
                    "prefix": filename.removesuffix(".html").split(".", 1)[1], "count": len(rows),
                    "statuses": dict(Counter(row["status"] for row in rows)),
                    "qualities": dict(Counter(row["quality"]["label"] for row in rows)),
                    "operations": sorted({str(row["params"].get("op", "")) for row in rows}),
                    "types": sorted({str(row["params"].get("dtype", row["params"].get("format", ""))) for row in rows})}
            pages.append(page)
            navigation = {"family_prefix_lengths": widths, "index": "../" + destination.name,
                          "position": position + 1, "pages": len(groups),
                          "previous": filenames[position - 1] if position else None,
                          "next": filenames[position + 1] if position + 1 < len(filenames) else None}
            payload = json.dumps({"overview": overview, "rows": rows, "pairs": local_pairs,
                                  "navigation": navigation}, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
            (detail_directory / filename).write_text(REPORT_TEMPLATE.replace("PAYLOAD", payload), encoding="utf-8")
    overview["measurement_quality_counts"] = dict(quality_counts)
    overview["implementation_counts"] = dict(implementation_counts)
    manifest = {"schema_version": 1, "layout": "offline_html_shards", "overview": overview,
                "family_prefix_lengths": widths, "detail_directory": detail_directory.name,
                "maximum_rows_per_page": REPORT_SHARD_ROWS, "pages": pages,
                "scope": "all selected cases; each detail shard is self-contained and needs no server or network"}
    destination.write_text(_shard_index_document(manifest), encoding="utf-8")
    atomic_json(destination.with_name(destination.stem + "_index.json"), manifest)
    atomic_json(directory / "coverage.json", overview)
    atomic_json(directory / "pairs.json", pairs)
    return destination


def _shard_index_document(manifest):
    payload = json.dumps(manifest, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return SHARD_INDEX_TEMPLATE.replace("PAYLOAD", payload)


def comparison_basis(reference, candidate):
    """A latency ratio is valid only when both modeled work and bytes match."""
    if not reference or not candidate or not is_comparable(reference) or not is_comparable(candidate):
        return {"basis": "unavailable", "speedup": None, "reason": "both cases need eligible measurements"}
    rw, cw = _number(reference.get("work_per_iteration")), _number(candidate.get("work_per_iteration"))
    rb, cb = _number(reference.get("bytes_per_iteration")), _number(candidate.get("bytes_per_iteration"))
    rt, ct = statistics.median(reference["samples_ms"]), statistics.median(candidate["samples_ms"])
    same_domain = _work_domain(reference)[0] == _work_domain(candidate)[0]
    multiplier = 1
    if all(value is not None and value >= 0 for value in (rw, cw, rb, cb)) and rw == cw and rb == cb and (rw == 0 or same_domain):
        basis, reason = "equal_work_latency", "modeled useful work and logical byte counts match"
    elif rw is not None and cw is not None and rw > 0 and cw > 0 and same_domain:
        basis, multiplier = "useful_work_rate", cw / rw
        reason = "work/byte counts differ; compare useful work per time rather than raw duration"
    elif rw == cw == 0 and rb is not None and cb is not None and rb > 0 and cb > 0:
        basis, multiplier = "logical_bandwidth", cb / rb
        reason = "payload counts differ; compare modeled useful byte rate rather than raw duration"
    else:
        return {"basis": "unavailable", "speedup": None, "reason": "missing or incompatible work models; no latency normalization assumed"}
    rstats, cstats = describe(reference["samples_ms"]), describe(candidate["samples_ms"])
    ri, ci = rstats.get("median_ci95_ms", [None, None]), cstats.get("median_ci95_ms", [None, None])
    bounds = ([1., 1.] if reference is candidate else
              [multiplier * ri[0] / ci[1], multiplier * ri[1] / ci[0]] if all(value is not None for value in ri + ci) else [None, None])
    return {"basis": basis, "speedup": multiplier * rt / ct, "reason": reason,
            "reference_work": rw, "candidate_work": cw, "reference_bytes": rb, "candidate_bytes": cb,
            "reference_timing_scope": reference.get("metadata", {}).get("timing_scope", reference.get("metadata", {}).get("timing_sample_scope")),
            "candidate_timing_scope": candidate.get("metadata", {}).get("timing_scope", candidate.get("metadata", {}).get("timing_sample_scope")),
            "endpoint_bounds": bounds,
            "bounds_scope": "endpoints from separate median intervals; not a jointly calibrated 95% speedup interval"}


def compare(directories: list[Path], destination: Path, *, allow_software_mismatch=False):
    if len(directories) < 2: raise ValueError("compare requires at least two run directories")
    loaded = [read_run(path) for path in directories]
    runs = [value[0] for value in loaded]
    results = [value[1] for value in loaded]
    plans = [read_plan(path, run) for path, run in zip(directories, runs)]
    def policy(run):
        identity = run["identity"]
        return {"policy": {k: v for k, v in identity["policy"].items() if k != "device"},
                "environment": {k: v for k, v in identity["environment"].items() if k not in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}}
    if any(policy(run) != policy(runs[0]) for run in runs[1:]):
        raise ValueError("run measurement policies/math environment differ; regenerate matched runs")
    software = [run["identity"].get("software", {}) for run in runs]
    mismatch = any(value != software[0] for value in software[1:])
    if mismatch and not allow_software_mismatch:
        raise ValueError("source/build/compiler/runtime/driver/library provenance differs; use matched software or explicitly --allow-software-mismatch")
    combined_plan = {key: case for plan in plans for key, case in plan.items()}
    names = [f"{path.name} · {run.get('device', {}).get('device_name', f'Run {i}')}"
             for i, (path, run) in enumerate(zip(directories, runs))]
    if len(combined_plan) > SINGLE_REPORT_LIMIT:
        return _compare_sharded(directories, destination, results, runs, plans, names, mismatch, combined_plan)
    rows = [_comparison_row(key, results, runs, plans) for key in sorted(combined_plan)]
    destination.mkdir(parents=True, exist_ok=True)
    result = {"baseline": str(directories[0]), "runs": runs, "rows": rows,
              "software_mismatch": mismatch,
              "warning": "Eligible exact-case ratios use equal-work latency or explicit work/byte-rate normalization; quality is separate; no aggregate score."}
    atomic_json(destination / "comparison.json", result)
    with (destination / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "family", "params"] + [f"run{i}_median_ms" for i in range(len(runs))] + [f"run{i}_speedup" for i in range(len(runs))]
                        + [f"run{i}_{field}" for field in ("median_us", "basis", "quality", "tflops", "integer_tops") for i in range(len(runs))])
        for row in rows:
            writer.writerow([row["id"], row["case"]["family"], json.dumps(row["case"]["params"], sort_keys=True), *row["median_ms"], *row["speedup_vs_first"],
                             *row["median_us"], *(b["basis"] for b in row["comparison_basis"]),
                             *(q["label"] if q else None for q in row["quality"]), *row["tflops"], *row["integer_tops"]])
    names = [f"{path.name} · {run.get('device', {}).get('device_name', f'Run {i}')}"
             for i, (path, run) in enumerate(zip(directories, runs))]
    document = _comparison_document(result, names)
    (destination / "comparison.html").write_text(document, encoding="utf-8")
    return destination / "comparison.html"



def _comparison_row(key, results, runs, plans):
    records = [values.get(key) for values in results]
    reference = records[0]
    medians = [statistics.median(row["samples_ms"]) if row and is_comparable(row) else None for row in records]
    bases = [comparison_basis(reference, row) for row in records]
    displays = [_display_row(row, runs[index].get("identity", {}).get("policy", {})) if row else None
                for index, row in enumerate(records)]
    return {"id": key, "case": next(plan[key].as_dict() for plan in plans if key in plan),
                 "median_ms": medians,
                 "median_us": [value * 1000 if value is not None else None for value in medians],
                 "speedup_vs_first": [value["speedup"] for value in bases],
                 "comparison_basis": bases,
                 "quality": [row["quality"] if row else None for row in displays],
                 "tflops": [row["tflops"] if row else None for row in displays],
                 "integer_tops": [row["integer_tops"] if row else None for row in displays],
                 "measurements": [{"display": row["display"], "stats": row["stats"],
                     "capability": row["capability"], "implementation": row["implementation"],
                     "samples_ms": row["samples_ms"], "validation": row["validation"],
                     "max_abs_error": row["max_abs_error"], "max_rel_error": row["max_rel_error"],
                     "work_per_iteration": row["work_per_iteration"], "bytes_per_iteration": row["bytes_per_iteration"],
                     "timing_scope": row["metadata"].get("timing_scope", row["metadata"].get("timing_sample_scope")),
                     "reason": row["reason"]} if row else None for row in displays],
                 "statuses": [row["status"] if row else "missing" for row in records],
                 "execution": [_visible_execution(row) if row else None for row in records],
                 "recorded_execution": [row.get("execution") if row else None for row in records]}


def _compare_sharded(directories, destination, results, runs, plans, names, mismatch, combined_plan):
    destination.mkdir(parents=True, exist_ok=True)
    detail_directory = destination / "comparison_details"
    detail_directory.mkdir(exist_ok=True)
    widths, groups = _partition_cases(combined_plan)
    common = {"baseline": str(directories[0]), "runs": runs, "software_mismatch": mismatch,
              "warning": "Exact-case ratios use equal-work latency or explicit work/byte-rate normalization; no aggregate score."}
    pages, filenames = [], list(groups)
    with (destination / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "family", "params"] + [f"run{i}_median_ms" for i in range(len(runs))] + [f"run{i}_speedup" for i in range(len(runs))]
                        + [f"run{i}_{field}" for field in ("median_us", "basis", "quality", "tflops", "integer_tops") for i in range(len(runs))])
        for position, (filename, identifiers) in enumerate(groups.items()):
            rows = [_comparison_row(key, results, runs, plans) for key in identifiers]
            for row in rows:
                writer.writerow([row["id"], row["case"]["family"], json.dumps(row["case"]["params"], sort_keys=True), *row["median_ms"], *row["speedup_vs_first"],
                                 *row["median_us"], *(b["basis"] for b in row["comparison_basis"]),
                                 *(q["label"] if q else None for q in row["quality"]), *row["tflops"], *row["integer_tops"]])
            navigation = {"index": "../comparison.html", "position": position + 1, "pages": len(groups),
                          "previous": filenames[position - 1] if position else None,
                          "next": filenames[position + 1] if position + 1 < len(filenames) else None}
            result = {**common, "rows": rows, "navigation": navigation}
            atomic_json(detail_directory / (filename.removesuffix(".html") + ".json"), result)
            (detail_directory / filename).write_text(_comparison_document(result, names), encoding="utf-8")
            pages.append({"file": f"comparison_details/{filename}", "family": combined_plan[identifiers[0]].family,
                          "prefix": filename.removesuffix(".html").split(".", 1)[1], "count": len(rows),
                          "statuses": dict(Counter(status for row in rows for status in row["statuses"])),
                          "qualities": dict(Counter(quality["label"] if quality else "not_measured" for row in rows for quality in row["quality"])),
                          "operations": sorted({str(row["case"]["params"].get("op", "")) for row in rows}),
                          "types": sorted({str(row["case"]["params"].get("dtype", row["case"]["params"].get("format", ""))) for row in rows})})
    manifest = {**common, "schema_version": 1, "layout": "offline_html_shards", "comparison": True,
                "selected_cases": len(combined_plan), "family_prefix_lengths": widths,
                "detail_directory": "comparison_details", "maximum_rows_per_page": REPORT_SHARD_ROWS,
                "pages": pages, "scope": "union of all selected plans; comparison rows reside in the listed HTML and sibling JSON shards"}
    atomic_json(destination / "comparison.json", manifest)
    (destination / "comparison.html").write_text(_shard_index_document(manifest), encoding="utf-8")
    return destination / "comparison.html"


def _comparison_document(result, names):
    payload = json.dumps({**result, "names": names}, allow_nan=False).replace("<", "\\u003c")
    template = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ArchBench comparison</title><style>
body{font:14px system-ui,sans-serif;margin:24px;background:#10151e;color:#eef4ff}p{line-height:1.65;max-width:1200px;color:#bac9dc}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:12px;border:1px solid #344257;vertical-align:top}th{background:#213149}tr:nth-child(even){background:#192231}input,button,select{padding:10px;background:#213149;border:1px solid #59708d;color:#fff;border-radius:5px}input[type=search]{width:min(600px,95%)}input[type=number]{width:80px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:18px 0}.table{overflow:auto}.metric{color:#69dac4;margin-bottom:6px;font-weight:600}.bad{color:#ffadad}small{display:block;color:#bac9dc;margin-top:5px}pre{white-space:pre-wrap;word-break:break-word}details{margin:20px 0;padding:15px;border:1px solid #344257}summary{cursor:pointer}#selected{padding:20px;background:#192231;border-radius:8px}
</style><h1>Exact-case GPU comparison</h1><p id="provenance-status"></p><p>Times use µs consistently. Useful floating arithmetic is TFLOP/s; integer arithmetic is TOP/s. Ratios compare each run with the first run: larger is faster. Equal work and byte counts permit a duration ratio. Different counts use an explicitly labeled useful-work or logical-byte rate ratio. Missing, unsupported and failed cases have no ratio. Quality describes within-run timing variation and is separate from correctness. There is no aggregate score.</p>
<div class="controls"><input id="search" type="search" placeholder="Search case IDs, types, shapes, operations or outcomes" aria-label="Search comparison"><button id="previous">Previous</button><label>Page <input id="page" type="number" min="1" value="1"></label><span id="page-total"></span><button id="next">Next</button><label>Rows <select id="size"><option>50</option><option selected>100</option><option>250</option></select></label><span id="count"></span></div>
<div class="table"><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div><section id="selected" hidden><button id="close">Close details</button><h2 id="selected-title"></h2><pre id="selected-content"></pre></section><details><summary>Comparison provenance and limitations</summary><pre id="provenance"></pre></details><script id="comparison-data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('comparison-data').textContent),$=id=>document.getElementById(id),fmt=v=>typeof v==='number'&&Number.isFinite(v)?Number(v.toPrecision(6)).toString():'N/A';
const text=(el,v)=>el.textContent=String(v??'');text($('provenance-status'),data.software_mismatch?'WARNING: source/build/toolkit/driver/library provenance differs across these runs.':'Software provenance matches; device architectures and hardware can still differ.');
if(data.navigation){const nav=document.createElement('p');for(const [label,href] of [['All comparisons',data.navigation.index],['Previous detail page',data.navigation.previous],['Next detail page',data.navigation.next]])if(href){const link=document.createElement('a');link.href=href;link.style.marginRight='18px';text(link,label);nav.append(link);}const scope=document.createElement('span');text(scope,`Detail page ${data.navigation.position}/${data.navigation.pages}`);nav.append(scope);document.body.prepend(nav);}
for(const title of ['Family / case','Exact parameters',...data.names,'Details']){const th=document.createElement('th');text(th,title);$('head').append(th);}
text($('provenance'),JSON.stringify({baseline:data.baseline,runs:data.runs,warning:data.warning,ratio_bounds:'Endpoints from separate median intervals are not a jointly calibrated 95% ratio confidence interval.',scope:'Requested case equality does not establish identical physical work when launch defaults depend on hardware.'},null,2));
for(const row of data.rows)row.search=[row.id,row.case.family,JSON.stringify(row.case.params),...row.statuses,...row.execution].join(' ').toLowerCase();let page=0;
function inspect(row){$('selected').hidden=false;text($('selected-title'),`${row.case.family} · ${row.id}`);text($('selected-content'),JSON.stringify(row,null,2));history.replaceState(null,'',`#case=${encodeURIComponent(row.id)}`);$('selected').scrollIntoView({behavior:'smooth'});}
function render(){const q=$('search').value.toLowerCase(),rows=data.rows.filter(row=>row.search.includes(q)),size=Number($('size').value),pages=Math.max(1,Math.ceil(rows.length/size));page=Math.max(0,Math.min(page,pages-1));$('page').value=String(page+1);$('page').max=String(pages);text($('page-total'),`of ${pages}`);text($('count'),`${rows.length} matching cases of ${data.rows.length}`);$('previous').disabled=page===0;$('next').disabled=page===pages-1;$('body').replaceChildren();
 for(const row of rows.slice(page*size,(page+1)*size)){const tr=document.createElement('tr');for(const value of [`${row.case.family} · ${row.id}`,Object.entries(row.case.params).map(([k,v])=>`${k}=${v}`).join(' ')]){const td=document.createElement('td');text(td,value);tr.append(td);}
 row.measurements.forEach((measurement,i)=>{const td=document.createElement('td'),primary=document.createElement('div');primary.className='metric';text(primary,measurement?measurement.display.median_time:'N/A');td.append(primary);const rate=document.createElement('div');text(rate,measurement?`${measurement.display.useful_rate} · ${measurement.display.logical_rate}`:'N/A');td.append(rate);const ratio=document.createElement('div');text(ratio,row.speedup_vs_first[i]==null?`No ratio · ${row.statuses[i]}`:`${fmt(row.speedup_vs_first[i])}× · ${row.comparison_basis[i].basis}`);td.append(ratio);const note=document.createElement('small');text(note,`${row.statuses[i]} · ${row.quality[i]?.label||'not_measured'}${row.quality[i]?.observed_cv==null?'':` · CV ${fmt(100*row.quality[i].observed_cv)}%`} · ${row.execution[i]||'not executed'}`);td.append(note);if(!measurement||row.speedup_vs_first[i]==null)td.className='bad';tr.append(td);});
 const td=document.createElement('td'),button=document.createElement('button');text(button,'Inspect');button.onclick=()=>inspect(row);td.append(button);tr.append(td);$('body').append(tr);}}
$('search').oninput=()=>{page=0;render();};$('size').onchange=()=>{page=0;render();};$('previous').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};$('page').onchange=()=>{page=Number($('page').value)-1;if(!Number.isFinite(page))page=0;render();};$('close').onclick=()=>{$('selected').hidden=true;history.replaceState(null,'',location.pathname+location.search);};render();if(location.hash.startsWith('#case=')){try{const selected=data.rows.find(row=>row.id===decodeURIComponent(location.hash.slice(6)));if(selected)inspect(selected);}catch(error){/* malformed hash */}}
</script></html>'''
    return template.replace("PAYLOAD", payload)



def _csv_writer(stream):
    writer = csv.DictWriter(stream, fieldnames=["id", "family", "params", "status", "validation", "execution",
        "median_ms", "useful_gops", "logical_gbytes_s", "reason", "unit", "useful_rate_unit",
        "work_per_iteration", "bytes_per_iteration", "max_abs_error", "max_rel_error",
        "abs_error_available", "rel_error_available", "recorded_execution", "error_note",
        "sample_count", "mean_ms", "min_ms", "max_ms", "mad_ms", "p10_batch_ms", "p90_batch_ms",
        "cv", "median_ci95_low_ms", "median_ci95_high_ms", "median_us", "median_ns", "stddev_ms",
        "mean_us", "min_us", "max_us", "mad_us", "stddev_us", "p10_batch_us", "p90_batch_us",
        "median_ci_method", "median_ci_target_coverage", "median_ci_achieved_coverage", "median_ci_finite",
        "half_run_drift_fraction", "quality", "quality_reason", "quality_target_cv", "quality_threshold_source", "quality_warnings", "capability", "support_scope",
        "implementation_class", "work_domain", "work_domain_source", "tflops", "integer_tops",
        "logical_gibytes_s", "useful_work_per_ns", "ns_per_useful_work", "useful_work_per_logical_byte",
        "useful_tera_rate", "useful_tera_rate_unit", "estimated_fft_tflops",
        "useful_rate_ci_low", "useful_rate_ci_high", "useful_tera_rate_ci_low", "useful_tera_rate_ci_high",
        "logical_gbytes_ci_low", "logical_gbytes_ci_high"])
    writer.writeheader()
    return writer


def _write_csv_row(writer, row):
    interval = row["stats"].get("median_ci95_ms", (None, None))
    values = {**row["stats"], **row, "params": json.dumps(row["params"], sort_keys=True),
              "sample_count": row["stats"].get("count"),
              "median_ci95_low_ms": interval[0], "median_ci95_high_ms": interval[1],
              "quality": row["quality"]["label"], "quality_reason": row["quality"]["reason"],
              "quality_target_cv": row["quality"]["target_cv"], "quality_threshold_source": row["quality"]["threshold_source"],
              "quality_warnings": "; ".join(row["quality"]["warnings"]),
              "capability": row["capability"]["availability"], "support_scope": row["capability"]["scope"],
              "implementation_class": row["implementation"]["class"]}
    for rate_key, prefix in (("useful_giga", "useful_rate"), ("useful_tera", "useful_tera_rate"),
                             ("logical_gbytes", "logical_gbytes")):
        bounds = (row["rates"].get(rate_key) or {}).get("ci95", [None, None])
        values[prefix + "_ci_low"], values[prefix + "_ci_high"] = bounds
    # DictWriter renders None as an empty CSV cell; it must not become zero.
    writer.writerow({key: values.get(key) for key in writer.fieldnames})


REPORT_TEMPLATE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ArchBench: measurements and implementation paths</title><style>
:root{color-scheme:dark;--bg:#10151e;--panel:#192231;--muted:#adbbce;--accent:#69dac4;--border:#344257}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eef4ff;font:14px system-ui,sans-serif}header,main{max-width:1800px;margin:auto;padding:24px}h1{margin:0 0 8px}h2{font-size:20px}p{line-height:1.6;color:var(--muted)}.cards{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}.card{background:var(--panel);padding:16px;border-radius:10px;min-width:160px}.number{font-size:30px;color:var(--accent)}.filters,.pager{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:14px 0}input,select,button{background:#253247;color:#fff;border:1px solid #52647d;padding:9px;border-radius:6px}input[type=search]{width:min(520px,95vw)}input[type=number]{width:80px}button{cursor:pointer}button:disabled{cursor:default;opacity:.4}button:hover:not(:disabled){border-color:var(--accent)}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:10px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}th{background:var(--panel);white-space:nowrap}th[data-key]{cursor:pointer}tr:hover{background:#1d293a}.ok{color:#69dac4}.bad{color:#ffadad}.neutral{color:var(--muted)}.table{overflow:auto}small{color:var(--muted)}details{margin:12px 0;border:1px solid var(--border);border-radius:8px;padding:12px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;word-break:break-word;line-height:1.5}.case-details{scroll-margin-top:20px;background:var(--panel);padding:20px;border-radius:10px;margin:24px 0}a{color:var(--accent)}.parameters{max-width:400px;overflow-wrap:anywhere}.nowrap{white-space:nowrap}svg{width:100%;max-width:1000px;height:auto;background:#111a27;border:1px solid var(--border);border-radius:6px}.legend{font-size:13px}.pager label{color:var(--muted)}
</style><header><h1>ArchBench</h1><div id="device"></div><p>Median times use µs consistently. Floating arithmetic uses useful TFLOP/s, integer arithmetic uses TOP/s, and other work retains its own units. Correctness, availability, and measurement quality are separate. N/A marks an undefined or uncomputed value; an exact measured error of zero remains 0.</p><div class="cards" id="cards"></div><p class="legend">Samples are retained per-call averages of batches where stated. Logical bandwidth models useful bytes, not measured DRAM transactions. Stability describes within-run dispersion, not guaranteed behavior on a later run. CSV times remain in milliseconds, with µs and ns columns also provided.</p></header><main>
<div class="filters"><input id="search" type="search" aria-label="Search cases" placeholder="Search ID, shape, type, operation, implementation or reason"><select id="family" aria-label="Family"><option value="">All families</option></select><select id="status" aria-label="Outcome"><option value="">All outcomes</option></select><select id="implementation" aria-label="Implementation"><option value="">All implementation classes</option></select><select id="quality" aria-label="Measurement quality"><option value="">All measurement qualities</option></select></div>
<div class="pager"><button id="previous">Previous</button><label>Page <input id="page" type="number" min="1" value="1"></label><span id="page-total"></span><button id="next">Next</button><label>Rows <select id="page-size"><option>50</option><option selected>100</option><option>250</option><option>1000</option></select></label><span id="count"></span></div>
<div class="table"><table><thead><tr><th data-key="family">Family / case</th><th>Parameters</th><th data-key="status">Outcome / validation</th><th data-key="capability">Availability</th><th data-key="implementation">Implementation</th><th data-key="quality">Measurement quality</th><th data-key="time">Median time (µs)</th><th data-key="rate">Useful rate (unit shown)</th><th data-key="bandwidth">Logical bandwidth</th><th>Median interval</th><th>Details</th></tr></thead><tbody id="body"></tbody></table></div>
<section id="case-details" class="case-details" hidden aria-label="Selected case details"><button id="close-details">Close details</button><h2 id="case-title"></h2><div id="pair-links" class="filters"></div><div id="sample-chart"></div><div id="case-content"></div></section>
<details><summary>Run provenance, finite coverage and measurement policy</summary><pre id="provenance"></pre></details><details><summary>Interpreting the measurements</summary><p>TFLOP/s counts useful floating operations, not issued instructions. Software arithmetic can implement the same numerical request with additional instructions. A native PTX request or accepted library API does not prove which SASS ran. Unsupported means unavailable for the recorded backend and environment, not globally impossible. Every case is reachable through pagination or search; no rows are discarded.</p><p>The default median interval uses binomial order-statistic coverage and records its achieved fixed-count coverage. Fewer than six samples cannot give a finite distribution-free 95% median interval. CV-based adaptive stopping is not corrected; nominal fixed-count coverage is not a guaranteed coverage for a stopped experiment. Independence and identical-distribution assumptions may fail under drift or shared-resource interference. Rate intervals invert timing bounds. Pair comparisons account for useful work counts and expose their timing scopes. Nanoseconds per useful operation is amortized reciprocal throughput, not instruction latency.</p></details>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), text=(el,value)=>el.textContent=String(value??'');
const fmt=value=>typeof value==='number'&&Number.isFinite(value)?Number(value.toPrecision(6)).toString():'N/A';
const overview=data.overview, device=overview.run.device||{}, byId=new Map(data.rows.map(row=>[row.id,row])),pairsById=new Map();
for(const pair of data.pairs.rows)for(const id of [pair.requested_case_id,pair.software_case_id]){if(!pairsById.has(id))pairsById.set(id,[]);pairsById.get(id).push(pair);}
if(data.navigation){const nav=document.createElement('p');for(const [label,href] of [['All cases index',data.navigation.index],['Previous detail page',data.navigation.previous],['Next detail page',data.navigation.next]])if(href){const link=document.createElement('a');link.href=href;text(link,label);link.style.marginRight='18px';nav.append(link);}const scope=document.createElement('span');text(scope,`Detail page ${data.navigation.position}/${data.navigation.pages} · ${data.rows.length} cases on this page`);nav.append(scope);document.querySelector('header').prepend(nav);}
text($('device'),`${device.device_name||'Unknown GPU'} · CC ${device.compute_capability||'?'} · ${device.binary_target||'?'} · runtime ${device.runtime_version||'?'}`);
for(const [label,value] of [['Selected cases',overview.selected],['Validated measurements',overview.validated_measured],[data.navigation?'Stable sets on this page':'Stable sample sets',data.rows.filter(row=>row.quality.label==='stable').length],['Requested / software pairs',data.pairs.counts.requested_implementations||0],['Requested pairs covered numerically',data.pairs.counts.numerical_problem_has_validated_implementation||0],['Unique paired problems covered',data.pairs.unique_operation_counts?.numerically_covered||0],['Recorded cases',overview.recorded],['Missing cases',overview.missing]]){
 const card=document.createElement('div');card.className='card';const number=document.createElement('div');number.className='number';text(number,value);card.append(number,document.createTextNode(label));$('cards').append(card);
}
for(const [id,values] of [['family',data.rows.map(r=>r.family)],['status',data.rows.map(r=>r.status)],['implementation',data.rows.map(r=>r.implementation.class)],['quality',data.rows.map(r=>r.quality.label)]]){
 for(const value of [...new Set(values)].sort()){const option=document.createElement('option');option.value=value;text(option,value);$(id).append(option);}
}
text($('provenance'),JSON.stringify(overview,null,2));
for(const row of data.rows)row.search_text=[row.id,row.family,JSON.stringify(row.params),row.status,row.validation,row.implementation.label,row.implementation.class,row.capability.scope,row.quality.label,row.notes].join(' ').toLowerCase();
let sortKey='family',direction=1,page=0;
function sortValue(row,key){if(key==='time')return row.stats.median_us??Infinity;if(key==='rate')return row.useful_gops??-Infinity;if(key==='bandwidth')return row.logical_gbytes_s??-Infinity;if(key==='quality')return row.stats.cv??Infinity;if(key==='capability')return row.capability.availability;if(key==='implementation')return row.implementation.class;return row[key];}
function interval(row){const ci=row.stats.median_ci95_us;if(ci&&ci.every(value=>typeof value==='number'))return `${fmt(ci[0])}–${fmt(ci[1])} µs`;return row.samples_ms.length?'No finite 95% CI':'N/A';}
function render(){
 const query=$('search').value.trim().toLowerCase(),family=$('family').value,status=$('status').value,implementation=$('implementation').value,quality=$('quality').value;
 const rows=data.rows.filter(row=>(!query||row.search_text.includes(query))&&(!family||row.family===family)&&(!status||row.status===status)&&(!implementation||row.implementation.class===implementation)&&(!quality||row.quality.label===quality));
 rows.sort((a,b)=>{const av=sortValue(a,sortKey),bv=sortValue(b,sortKey);return(av<bv?-1:av>bv?1:0)*direction||a.id.localeCompare(b.id);});
 const size=Number($('page-size').value),pages=Math.max(1,Math.ceil(rows.length/size));page=Math.min(Math.max(page,0),pages-1);const start=page*size;
 $('page').value=String(page+1);$('page').max=String(pages);text($('page-total'),`of ${pages}`);$('previous').disabled=page===0;$('next').disabled=page===pages-1;text($('count'),`${rows.length?start+1:0}–${Math.min(start+size,rows.length)} of ${rows.length} matching cases (${overview.recorded} recorded, ${overview.missing} awaiting results)`);$('body').replaceChildren();
 for(const row of rows.slice(start,start+size)){
  const tr=document.createElement('tr');const qualityText=`${row.quality.label}${row.stats.cv==null?'':` · CV ${fmt(100*row.stats.cv)}%`}`;
  const values=[`${row.family} · ${row.id}`,Object.entries(row.params).map(([key,value])=>`${key}=${value}`).join(' '),`${row.status} / ${row.validation}`,`${row.capability.availability} (${row.capability.scope})`,row.implementation.label,qualityText,row.display.median_time,row.display.useful_rate,row.display.logical_rate,interval(row)];
  values.forEach((value,index)=>{const td=document.createElement('td');text(td,value);if(index===1)td.className='parameters';if(index===2)td.className=row.status==='ok'?'ok':row.status==='measurement_only'?'neutral':'bad';if(index===5)td.className=row.quality.label==='stable'?'ok':row.quality.label==='variable'?'bad':'neutral';if(index===6||index===7||index===8)td.classList.add('nowrap');tr.append(td);});
  const td=document.createElement('td'),button=document.createElement('button');text(button,'Inspect');button.onclick=()=>showCase(row.id);td.append(button);tr.append(td);$('body').append(tr);
 }
}
function drawSamples(row){
 const target=$('sample-chart');target.replaceChildren();if(!row.samples_us.length)return;
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 1000 180');svg.setAttribute('role','img');svg.setAttribute('aria-label','Retained timing samples in microseconds');
 const minimum=Math.min(...row.samples_us),maximum=Math.max(...row.samples_us),spread=maximum-minimum||Math.max(maximum*.02,1e-9),points=[];
 row.samples_us.forEach((value,index)=>{const x=65+(row.samples_us.length===1?435:index*870/(row.samples_us.length-1)),y=140-(value-minimum)*105/spread;points.push(`${x},${y}`);const point=document.createElementNS(ns,'circle');point.setAttribute('cx',x);point.setAttribute('cy',y);point.setAttribute('r','3');point.setAttribute('fill','#69dac4');const title=document.createElementNS(ns,'title');title.textContent=`Sample ${index+1}: ${fmt(value)} µs`;point.append(title);svg.append(point);});
 const line=document.createElementNS(ns,'polyline');line.setAttribute('points',points.join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#69dac4');line.setAttribute('stroke-width','1.5');svg.prepend(line);
 for(const [x,y,value] of [[12,18,'Retained per-call sample averages (µs)'],[12,38,fmt(maximum)],[12,143,fmt(minimum)],[65,168,'1'],[920,168,String(row.samples_us.length)]]){const label=document.createElementNS(ns,'text');label.setAttribute('x',x);label.setAttribute('y',y);label.setAttribute('fill','#adbbce');label.setAttribute('font-size','12');label.textContent=value;svg.append(label);}target.append(svg);
}
function showCase(id){
 const row=byId.get(id);if(!row)return;const panel=$('case-details');panel.hidden=false;text($('case-title'),`${row.family} · ${row.id}`);$('case-content').replaceChildren();drawSamples(row);
 $('pair-links').replaceChildren();for(const peerId of row.pairing?.paired_case_ids||[]){const peer=byId.get(peerId);if(!peer){if(data.navigation){const [family,digest]=peerId.split('.'),width=data.navigation.family_prefix_lengths[family];if(width){const link=document.createElement('a');link.href=`${family}.${digest.slice(0,width)}.html#case=${encodeURIComponent(peerId)}`;text(link,`Inspect counterpart: ${peerId}`);$('pair-links').append(link);}}continue;}const button=document.createElement('button');text(button,`Inspect counterpart: ${peer.pairing?.requested_backend||peer.implementation.class} · ${peer.status}`);button.onclick=()=>showCase(peer.id);$('pair-links').append(button);}
 const sections=[['Parameters and work model',{params:row.params,work_domain:row.work_domain,work_domain_source:row.work_domain_source,original_unit:row.unit,work_per_iteration:row.work_per_iteration,bytes_per_iteration:row.bytes_per_iteration,pairing:row.pairing},true],['Timing, stability and confidence interval',{stats:row.stats,quality:row.quality,timing_phases:row.timing_phases},true],['Useful rates and rate uncertainty',row.rates,true],['Correctness and availability',{validation:row.validation,max_abs_error:row.max_abs_error,max_rel_error:row.max_rel_error,abs_error_available:row.abs_error_available,rel_error_available:row.rel_error_available,error_note:row.error_note,reason:row.reason,capability:row.capability,implementation:row.implementation},true],['Raw samples',{samples_ms:row.samples_ms,samples_us:row.samples_us},false],['Recorded metrics and provenance',{recorded_execution:row.recorded_execution,recorded_max_abs_error:row.recorded_max_abs_error,recorded_max_rel_error:row.recorded_max_rel_error,metrics:row.metrics,metadata:row.metadata},false]];
 if(pairsById.has(id))sections.splice(1,0,['Requested and software counterpart comparison',pairsById.get(id),true]);
 for(const [title,value,opened] of sections){const detail=document.createElement('details');detail.open=opened;const heading=document.createElement('summary'),pre=document.createElement('pre');text(heading,title);text(pre,JSON.stringify(value,null,2));detail.append(heading,pre);$('case-content').append(detail);}
 history.replaceState(null,'',`#case=${encodeURIComponent(id)}`);panel.scrollIntoView({behavior:'smooth',block:'start'});
}
for(const id of ['search','family','status','implementation','quality','page-size'])$(id).addEventListener(id==='search'?'input':'change',()=>{page=0;render();});
$('previous').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};$('page').onchange=()=>{page=Number($('page').value)-1;if(!Number.isFinite(page))page=0;render();};$('close-details').onclick=()=>{$('case-details').hidden=true;history.replaceState(null,'',location.pathname+location.search);};
for(const th of document.querySelectorAll('th[data-key]'))th.onclick=()=>{direction=sortKey===th.dataset.key?-direction:1;sortKey=th.dataset.key;page=0;render();};
render();if(location.hash.startsWith('#case=')){try{showCase(decodeURIComponent(location.hash.slice(6)));}catch(error){/* Invalid external hash does not invalidate the report. */}}
</script></html>'''


SHARD_INDEX_TEMPLATE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ArchBench large report index</title><style>
body{font:14px system-ui,sans-serif;background:#10151e;color:#eef4ff;margin:24px}h1{margin-bottom:8px}p{line-height:1.6;color:#bac9dc;max-width:1200px}a{color:#69dac4}.cards,.controls{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0;align-items:center}.card{padding:16px;background:#192231;border-radius:8px;min-width:140px}.number{font-size:28px;color:#69dac4}input,button,select{padding:10px;border:1px solid #59708d;border-radius:6px;background:#213149;color:#fff}input{width:min(540px,90vw)}button{cursor:pointer}button:disabled{opacity:.4}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:12px;border-bottom:1px solid #344257;text-align:left}th{background:#213149}tr:hover{background:#192231}pre{white-space:pre-wrap;word-break:break-word}details{margin:22px 0;border:1px solid #344257;padding:15px;border-radius:8px}summary{cursor:pointer}.table{overflow:auto}#page{width:80px}.error{color:#ffadad}
</style><h1>ArchBench — complete report index</h1><p id="description"></p><p>This index keeps the initial page small. Every selected case, including missing results, has a detail page. Detail pages are self-contained HTML files with local pagination, filters, raw timing samples, uncertainty, validation, implementation labels, and software counterpart links. Keep this index beside its detail folder; no web server or network is needed.</p><div class="cards" id="cards"></div>
<div class="controls"><input id="case-id" placeholder="Open exact case ID, for example gemm.0123456789abcdef0123" aria-label="Exact case ID"><button id="open-case">Open case</button><span id="lookup-error" class="error"></span></div><div class="controls"><input id="filter" type="search" placeholder="Filter page groups by family, dtype, operation or outcome" aria-label="Filter detail pages"><button id="previous">Previous</button><label>Page <input id="page" type="number" min="1" value="1"></label><span id="page-total"></span><button id="next">Next</button><span id="count"></span></div>
<div class="table"><table><thead><tr><th>Detail page</th><th>Cases</th><th>Types</th><th>Operations</th><th>Outcomes</th><th>Measurement quality</th></tr></thead><tbody id="body"></tbody></table></div><details><summary>Provenance, coverage, and layout</summary><pre id="provenance"></pre></details><script id="index-data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('index-data').textContent),$=id=>document.getElementById(id),text=(el,value)=>el.textContent=String(value??'');
const overview=data.overview||{},comparison=data.comparison===true; text($('description'),`${data.pages.length} bounded detail pages cover ${overview.selected??data.selected_cases} cases. ${comparison?'Comparison ratios state whether they use equal-work latency or normalized useful rates.':'Times use µs consistently; floating arithmetic uses TFLOP/s, integer arithmetic uses TOP/s, and nonarithmetic work keeps its own units.'} N/A is distinct from measured zero.`);
for(const [label,value] of [['Selected cases',overview.selected??data.selected_cases],['Recorded cases',overview.recorded??'See each run'],['Validated measurements',overview.validated_measured??'See each run'],['Stable sample sets',overview.measurement_quality_counts?.stable??'See details'],['Detail pages',data.pages.length]]){const card=document.createElement('div');card.className='card';const number=document.createElement('div');number.className='number';text(number,value);const caption=document.createElement('div');text(caption,label);card.append(number,caption);$('cards').append(card);}
text($('provenance'),JSON.stringify({...data,pages:`${data.pages.length} pages listed above`},null,2));
const files=new Set(data.pages.map(page=>page.file));for(const part of data.pages)part.search=[part.family,part.prefix,...part.types,...part.operations,...Object.keys(part.statuses),...Object.keys(part.qualities)].join(' ').toLowerCase();let page=0;
function render(){const query=$('filter').value.toLowerCase(),parts=data.pages.filter(part=>part.search.includes(query)),size=100,pages=Math.max(1,Math.ceil(parts.length/size));page=Math.max(0,Math.min(page,pages-1));$('page').value=String(page+1);$('page').max=String(pages);text($('page-total'),`of ${pages}`);text($('count'),`${parts.length} matching detail pages`);$('previous').disabled=page===0;$('next').disabled=page===pages-1;$('body').replaceChildren();for(const part of parts.slice(page*size,(page+1)*size)){const tr=document.createElement('tr'),linkCell=document.createElement('td'),link=document.createElement('a');link.href=part.file;text(link,`${part.family} · ${part.prefix}…`);linkCell.append(link);tr.append(linkCell);for(const value of [part.count,part.types.join(', '),part.operations.join(', '),JSON.stringify(part.statuses),JSON.stringify(part.qualities)]){const cell=document.createElement('td');text(cell,value);tr.append(cell);}$('body').append(tr);}}
function openCase(){const value=$('case-id').value.trim(),match=/^([a-z][a-z0-9_]*)\.([0-9a-f]{20})$/.exec(value),width=match&&data.family_prefix_lengths[match[1]];if(!width){text($('lookup-error'),'Enter a full case ID from a selected family.');return;}const file=`${data.detail_directory}/${match[1]}.${match[2].slice(0,width)}.html`;if(!files.has(file)){text($('lookup-error'),'No selected case group matches that ID.');return;}location.href=`${file}#case=${encodeURIComponent(value)}`;}
$('open-case').onclick=openCase;$('case-id').onkeydown=event=>{if(event.key==='Enter')openCase();};$('filter').oninput=()=>{page=0;render();};$('previous').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};$('page').onchange=()=>{page=Number($('page').value)-1;if(!Number.isFinite(page))page=0;render();};render();
</script></html>'''
