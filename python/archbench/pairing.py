# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit numerical counterparts; never relabel a rejected request as native.

An operation key groups requests that differ only by implementation backend and
software-policy. A mathematical-reference pair preserves the stored tensors and
mathematical problem, but does not claim to reproduce a library's internal
emulation strategy, fast accumulation, or instruction schedule.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
from .model import Case, canonical, is_measured_pass

DEFAULT_BACKENDS = {"tensor": "native", "gemm": "cublas", "blas": "cublas",
                    "fft": "cufft", "sparse": "cusparse", "quant_gemm": "cublas"}
REQUEST_BACKENDS = {"tensor": {"native", "ptx"}, "gemm": {"cublas", "cublaslt"},
                    "blas": {"cublas", "native"}, "fft": {"cufft", "native"},
                    "sparse": {"cusparse", "native"}, "quant_gemm": {"cublas"}}
SOFTWARE_BACKENDS = {"software", "emulated"}


def requested_backend(case: Case) -> str:
    default = "software" if case.family == "quant_gemm" and case.params.get("mode", "fused") == "fused" else DEFAULT_BACKENDS.get(case.family, "default")
    return str(case.params.get("backend", default))


def is_software_case(case: Case) -> bool:
    return case.family in DEFAULT_BACKENDS and requested_backend(case) in SOFTWARE_BACKENDS


def operation_id(case: Case) -> str:
    params = dict(case.params)
    params.pop("backend", None)
    params.pop("software-policy", None)
    # Preserve every other parameter, including compute policy, input pattern,
    # layout, dimensions and tuning requests. This intentionally avoids guessing
    # that superficially similar but different experiments are equivalent.
    value = {"family": case.family, "provider": case.provider, "params": params}
    return "operation." + hashlib.sha256(canonical(value).encode()).hexdigest()[:20]


def software_counterpart(case: Case) -> Case | None:
    if case.provider != "core" or case.family not in REQUEST_BACKENDS:
        return None
    if requested_backend(case) not in REQUEST_BACKENDS[case.family]:
        return None
    params = {**case.params, "backend": "emulated" if case.family == "tensor" else "software"}
    if case.family == "gemm":
        if params.get("compute") in {"fixedpoint", "bf16x9"} or params.get("fast-accum") in (True, "true", "1", 1):
            params["software-policy"] = "mathematical-reference"
    return Case(case.family, params, case.tags, case.provider)


def with_software_counterparts(cases):
    """Preserve requests and add real software cases, deduplicated by case ID."""
    seen = set()
    for case in cases:
        for candidate in (case, software_counterpart(case)):
            if candidate is not None and candidate.id not in seen:
                seen.add(candidate.id)
                yield candidate


def pairing_index(cases):
    index = defaultdict(list)
    for case in cases:
        index[operation_id(case)].append(case)
    return dict(index)


def pairing_metadata(case, index):
    key = operation_id(case)
    peers = index.get(key, [])
    counterpart = software_counterpart(case)
    software_id = case.id if is_software_case(case) else counterpart.id if counterpart else None
    source_ids = [item.id for item in peers if software_counterpart(item) is not None]
    relation = "mathematical_reference" if ((counterpart or case).params.get("software-policy") == "mathematical-reference") else "same_request"
    return {"schema_version": 1, "operation_id": key,
            "requested_backend": requested_backend(case),
            "role": "software" if is_software_case(case) else "requested",
            "relation": relation,
            "software_case_id": software_id,
            "software_pair_selected": software_id is not None and any(item.id == software_id for item in peers),
            "paired_case_ids": [item.id for item in peers if item.id != case.id],
            "source_case_ids": source_ids,
            "policy_model": "mathematical_reference" if relation == "mathematical_reference" else "same_requested_numeric_controls",
            "bitwise_equivalence_claim": False,
            "library_algorithm_equivalence_claim": False}


def pair_summary(cases, records, *, include_rows=True):
    """Account separately for requested implementations and numerical references."""
    cases = list(cases)
    selected = {case.id for case in cases}
    counts = Counter()
    operations = {}
    rows = []
    for case in cases:
        counterpart = software_counterpart(case)
        if counterpart is None:
            continue
        requested, software = records.get(case.id), records.get(counterpart.id)
        native_ok = bool(requested and is_measured_pass(requested))
        software_ok = bool(software and is_measured_pass(software))
        relation = "mathematical_reference" if counterpart.params.get("software-policy") == "mathematical-reference" else "same_request"
        state = operations.setdefault(operation_id(case), {"requested_validated": False, "software_validated": False})
        state["requested_validated"] |= native_ok
        state["software_validated"] |= software_ok
        counts["requested_implementations"] += 1
        counts["software_pairs_selected"] += counterpart.id in selected
        counts["software_pairs_not_selected"] += counterpart.id not in selected
        counts["both_validated" if native_ok and software_ok else "requested_only_validated" if native_ok
               else "software_only_validated" if software_ok else "neither_validated"] += 1
        counts["numerical_problem_has_validated_implementation"] += native_ok or software_ok
        counts[relation + "_pairs"] += 1
        if include_rows:
            rows.append({"operation_id": operation_id(case), "requested_case_id": case.id,
                         "software_case_id": counterpart.id, "relation": relation,
                         "requested_status": requested.get("status") if requested else "missing",
                         "software_status": software.get("status") if software else "missing" if counterpart.id in selected else "not_selected",
                         "requested_validated": native_ok, "software_validated": software_ok,
                         "numerical_problem_covered": native_ok or software_ok})
    distinct = Counter(selected=len(operations))
    for state in operations.values():
        requested_ok, software_ok = state["requested_validated"], state["software_validated"]
        distinct["both_validated" if requested_ok and software_ok else "requested_only_validated" if requested_ok
                 else "software_only_validated" if software_ok else "neither_validated"] += 1
        distinct["numerically_covered"] += requested_ok or software_ok
    return {"schema_version": 1, "counts": dict(counts), "counts_denominator": "requested implementations (one row per requested/software pair)",
            "unique_operation_counts": dict(distinct),
            "scope": "selected requests and explicit software counterparts; not every possible input or CUDA operation",
            "mathematical_reference_note": "A numerical reference does not establish support for the requested library algorithm or emulation policy.",
            "rows": rows if include_rows else []}
