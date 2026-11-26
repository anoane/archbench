#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate or verify README's complete benchmark-family catalogue."""
from __future__ import annotations

import argparse
import difflib
from functools import lru_cache
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from archbench.catalog import FAMILIES, generate  # noqa: E402
from archbench.pairing import (  # noqa: E402
    requested_backend,
    with_software_counterparts,
)


README = ROOT / "README.md"
START = "<!-- BEGIN GENERATED BENCHMARK CATALOG -->"
END = "<!-- END GENERATED BENCHMARK CATALOG -->"
TYPE_FIELDS = ("dtype", "dtype-b", "output-dtype", "accumulator", "format")
POLICY_FIELDS = (
    "mode",
    "precision",
    "round",
    "compute",
    "cublas-algo",
    "scale",
    "scale-exp-a",
    "scale-exp-b",
    "scale-mode",
    "scale-format",
    "mma-kind",
    "pattern",
    "reset-every",
    "epilogue",
    "direction",
    "trans-a",
    "trans-b",
    "batch-mode",
    "emulation-strategy",
    "mantissa-control",
    "mantissa-bits",
    "software-policy",
    "side",
    "uplo",
    "diag",
    "input-pattern",
    "distribution",
    "causal",
    "pinned",
)
FIELD_LABELS = {
    "dtype-b": "dtype-b",
    "output-dtype": "output",
    "scale-mode": "scale-mode",
    "scale-format": "scale-format",
    "input-pattern": "input",
}


def normalized(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.lstrip("-").isdigit() else part.lower()
        for part in re.split(r"(-?\d+)", value)
    )


def code(value: str) -> str:
    return f"`{value}`"


def compact_values(values: set[str]) -> str:
    """Render every value, compacting only complete contiguous numeric ranges."""
    if not values:
        return "—"
    groups: dict[str, list[tuple[int, str]]] = {}
    literals: list[str] = []
    for value in values:
        match = re.fullmatch(r"([^0-9]*)([0-9]+)", value)
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), value))
        else:
            literals.append(value)
    rendered: list[tuple[tuple[object, ...], str]] = []
    for prefix, members in groups.items():
        members.sort()
        numbers = [number for number, _ in members]
        if len(numbers) >= 4 and numbers == list(range(numbers[0], numbers[-1] + 1)):
            first = f"{prefix}{numbers[0]}"
            last = f"{prefix}{numbers[-1]}"
            rendered.append((natural_key(first), f"{code(first)}–{code(last)}"))
        else:
            rendered.extend((natural_key(value), code(value)) for _, value in members)
    rendered.extend((natural_key(value), code(value)) for value in literals)
    return ", ".join(value for _, value in sorted(rendered, key=lambda item: item[0]))


@lru_cache(maxsize=1)
def inventory() -> tuple[dict[str, dict[str, object]], int]:
    records = {
        family: {
            "count": 0,
            "ops": set(),
            "backends": set(),
            "fields": {field: set() for field in TYPE_FIELDS + POLICY_FIELDS},
            "bits": set(),
            "signed": set(),
        }
        for family in FAMILIES
    }
    total = 0
    for item in with_software_counterparts(generate("full")):
        total += 1
        record = records[item.family]
        record["count"] = int(record["count"]) + 1
        operation = item.params.get(
            "op", "gemm" if item.family.endswith("gemm") else "family-default"
        )
        record["ops"].add(normalized(operation))
        backend = requested_backend(item)
        record["backends"].add("core" if backend == "default" else backend)
        fields = record["fields"]
        for field in TYPE_FIELDS + POLICY_FIELDS:
            if field in item.params:
                fields[field].add(normalized(item.params[field]))
        if "bits" in item.params:
            record["bits"].add(normalized(item.params["bits"]))
        if "signed" in item.params:
            record["signed"].add(normalized(item.params["signed"]))
    return records, total


def labeled_fields(record: dict[str, object], names: tuple[str, ...]) -> str:
    fields = record["fields"]
    rendered = []
    for name in names:
        values = fields[name]
        if values:
            rendered.append(
                f"{FIELD_LABELS.get(name, name)}: {compact_values(values)}"
            )
    if names == TYPE_FIELDS and record["bits"]:
        rendered.append(f"bits: {compact_values(record['bits'])}")
        rendered.append(f"signed: {compact_values(record['signed'])}")
    return "; <br>".join(rendered) if rendered else "—"


@lru_cache(maxsize=1)
def render_block() -> str:
    records, total = inventory()
    lines = [
        START,
        "",
        "This table is generated from the effective `full` selection after default",
        "software-pair expansion. Contiguous integer and quantization widths are",
        "compressed as inclusive ranges; every other operation, type, backend and",
        "mode/policy value is written explicitly.",
        "",
        "| Family | Cases | Operations | Types and formats | Backends | Modes and policies |",
        "|---|---:|---|---|---|---|",
    ]
    for family in FAMILIES:
        record = records[family]
        lines.append(
            "| {family} | {count:,} | {ops} | {types} | {backends} | {policies} |".format(
                family=code(family),
                count=record["count"],
                ops=compact_values(record["ops"]),
                types=labeled_fields(record, TYPE_FIELDS),
                backends=compact_values(record["backends"]),
                policies=labeled_fields(record, POLICY_FIELDS),
            )
        )
    lines.extend(
        (
            f"| **Total** | **{total:,}** | **20 registered families** | — | — | — |",
            "",
            END,
        )
    )
    return "\n".join(lines)


def replaced_readme(text: str, block: str) -> str:
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("README must contain exactly one generated-catalog marker pair")
    start = text.index(START)
    end = text.index(END, start) + len(END)
    return text[:start] + block + text[end:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="fail if README differs")
    action.add_argument("--update", action="store_true", help="rewrite generated block")
    args = parser.parse_args(argv)
    original = README.read_text(encoding="utf-8")
    expected = replaced_readme(original, render_block())
    if args.update:
        if expected != original:
            README.write_text(expected, encoding="utf-8")
        print("README benchmark catalogue updated")
        return 0
    if expected == original:
        print("README benchmark catalogue is current")
        return 0
    difference = difflib.unified_diff(
        original.splitlines(),
        expected.splitlines(),
        fromfile="README.md",
        tofile="generated README.md",
        lineterm="",
    )
    sys.stderr.write("\n".join(difference) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
