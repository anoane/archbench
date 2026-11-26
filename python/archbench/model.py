# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stable case identity and conservative result accounting."""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any

SCHEMA_VERSION = 1
SUCCESS = {"ok"}
VALIDATED = {"passed", "pass", "passed_sampled", "passed_full"}
TERMINAL = {"ok", "validation_failed", "unsupported", "not_implemented", "not_validated",
            "missing_binary", "architecture_mismatch", "memory_budget", "out_of_memory",
            "timeout", "error", "interrupted", "compile_failed", "missing_dependency", "probe_only", "measurement_only"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Case:
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    provider: str = "core"

    def __post_init__(self):
        if not isinstance(self.family, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", self.family):
            raise ValueError("family must be a lowercase identifier")
        if not isinstance(self.params, dict): raise ValueError("case params must be an object")
        reserved = {"family", "info", "help", "device", "samples", "max-samples", "target-cv", "warmup", "warmup-ms", "repeats", "target-ms", "max-repeats", "memory-fraction", "memory-limit-mib"}
        for key, value in self.params.items():
            if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9-]*", key):
                raise ValueError("case option names must be simple identifiers")
            if key in reserved: raise ValueError(f"case cannot override runner-owned option {key}")
            if type(value) not in (str, bool, int, float): raise ValueError("case values must be scalar JSON values")
            if type(value) is float and not math.isfinite(value): raise ValueError("nonfinite case value")
        object.__setattr__(self, "params", dict(self.params))

    @property
    def id(self) -> str:
        # Device, architecture, and timing policy are run metadata, not mathematical identity.
        digest = hashlib.sha256(canonical({"family": self.family, "params": self.params,
                                          "provider": self.provider}).encode()).hexdigest()[:20]
        return f"{self.family}.{digest}"

    @property
    def label(self) -> str:
        return self.family + " " + " ".join(f"{k}={v}" for k, v in sorted(self.params.items()))

    def argv(self) -> list[str]:
        result = ["--family", self.family]
        for name, value in sorted(self.params.items()):
            result.extend(["--" + name, str(value).lower() if isinstance(value, bool) else str(value)])
        return result

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "family": self.family, "params": self.params,
                "tags": list(self.tags), "provider": self.provider}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Case":
        case = cls(value["family"], value.get("params", {}), tuple(value.get("tags", ())),
                   value.get("provider", "core"))
        if "id" in value and value["id"] != case.id:
            raise ValueError("case ID does not match its contents")
        return case


def is_measured_pass(record: dict[str, Any]) -> bool:
    samples = record.get("samples_ms", [])
    return (record.get("status") in SUCCESS and record.get("validation") in VALIDATED and
            bool(samples) and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples))


def is_comparable(record: dict[str, Any]) -> bool:
    if is_measured_pass(record): return True
    samples = record.get("samples_ms", [])
    return (record.get("status") == "measurement_only" and record.get("validation") == "not_applicable"
            and bool(samples) and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples))


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if record.get("status") not in TERMINAL:
        errors.append("unknown status")
    samples = record.get("samples_ms", [])
    if not isinstance(samples, list) or any(type(x) not in (int, float) or
            not math.isfinite(x) or x <= 0 for x in samples):
        errors.append("samples must be finite positive milliseconds")
    if record.get("status") == "ok" and not is_measured_pass(record):
        errors.append("ok requires numerical validation and positive timing samples")
    if record.get("status") == "measurement_only" and not is_comparable(record):
        errors.append("measurement_only requires positive samples and not_applicable validation")
    return errors
