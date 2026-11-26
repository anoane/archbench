#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reproduce a bounded ILP/code-generation audit of compiled ALU kernels.

This preserves exact cuobjdump commands, object hashes, disassembly and static
instruction evidence. It does not execute a GPU or infer dynamic counters.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


QUERIES = (
    ("80", "f16_tensor_f32_ilp8", "tensor_kernel", r"tensor_formatE0ELS\d+_0ELi0ELb0ELi8", "mma", 8),
    ("89", "fp8_tensor_f32_ilp8", "tensor_kernel", r"tensor_formatE13ELS\d+_13ELi0ELb0ELi8", "mma", 8),
    ("89", "fp8_tensor_f16_ilp8", "tensor_half_kernel", r"tensor_formatE13ELS\d+_13ELb0ELi8", "mma", 8),
    ("89", "fp32_scalar_fma_ilp8", "floating_kernelIf", r"float_opE3ELi8", "FFMA", 8),
    ("89", "fp16_packed_fma_ilp8", "packed_float_kernelI6__half", r"float_opE3ELi8", "HFMA2", 8),
    ("89", "fp8_emulated_f32_ilp8", "tensor_kernel", r"tensor_formatE13ELS\d+_13ELi0ELb1ELi8", "FFMA", 32),
    ("120a", "fp4_tensor_f32_ilp8", "tensor_kernel", r"tensor_formatE10ELS\d+_10ELi0ELb0ELi8", "mma", 8),
    ("120a", "int4_tensor_i32_ilp8", "tensor_kernel", r"tensor_formatE6ELS\d+_6ELi0ELb0ELi8", "expanded", 8),
    ("120a", "binary_and_tensor_i32_ilp8", "tensor_kernel", r"tensor_formatE8ELS\d+_8ELi0ELb0ELi8", "expanded", 8),
)
MATH = re.compile(r"\b((?:H|Q|I|D|B)MMA[^ ]*|[FD]FMA[^ ]*|HFMA2[^ ]*)\s+([^;]+)")
CALL = re.compile(r"\bCALL[^ ]*\s+(0x[0-9a-f]+)\s*;", re.IGNORECASE)


def capture(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True, help="CMake build directory containing worker objects")
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump"), help="path to the CUDA toolkit cuobjdump")
    parser.add_argument("--output", type=Path, help="evidence directory; default BUILD/kernel_audit")
    parser.add_argument("--arch", action="append", choices=("80", "89", "120a"), help="repeat to select targets; default all")
    args = parser.parse_args()
    if not args.cuobjdump:
        parser.error("cuobjdump is required; provide its CUDA toolkit path")
    build = args.build.resolve()
    output = (args.output or build / "kernel_audit").resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = set(args.arch or ("80", "89", "120a"))
    source = Path(__file__).resolve().parents[1] / "src/alu.cu"
    version = capture([args.cuobjdump, "--version"]).strip()
    symbols, reports = {}, []
    for arch, label, kind, pattern, expected_kind, expected_count in QUERIES:
        if arch not in targets:
            continue
        obj = build / f"CMakeFiles/archbench_sm{arch}.dir/src/alu.cu.o"
        report = {"target": arch, "kernel": label, "object": str(obj),
                  "scope": "compiled-kernel static SASS evidence; no GPU execution, dynamic counter, or performance claim"}
        if not obj.is_file():
            report.update(status="missing_object", reason="build this architecture worker first")
            reports.append(report)
            continue
        symbol_command = [args.cuobjdump, "--dump-elf-symbols", str(obj)]
        if arch not in symbols:
            symbols[arch] = capture(symbol_command)
        functions = [line.split()[-1] for line in symbols[arch].splitlines()
                     if line.startswith("FUNC") and kind in line and re.search(pattern, line)]
        report.update(object_sha256=sha256(obj), source_sha256_observed_at_audit=sha256(source),
                      source_newer_than_object=source.stat().st_mtime_ns > obj.stat().st_mtime_ns,
                      symbol_extraction_command=symbol_command, matched_symbols=functions)
        if len(functions) != 1:
            report.update(status="not_identified", reason="expected one matching kernel; it may be absent or use a changed mangling")
            reports.append(report)
            continue
        command = [args.cuobjdump, "--dump-sass", "--function", functions[0], str(obj)]
        sass = capture(command)
        path = output / f"sm{arch}_{label}.sass"
        path.write_text(sass)
        math = {}
        for line in sass.splitlines():
            match = MATH.search(line)
            if match:
                math.setdefault(match.group(1), []).append(match.group(2).strip())
        calls = Counter(CALL.findall(sass))
        relevant = [operands for opcode, instructions in math.items()
                    if ("MMA" in opcode if expected_kind in ("mma", "expanded") else opcode.startswith(expected_kind))
                    for operands in instructions]
        destinations = sorted({operands.split(",")[0] for operands in relevant})
        if expected_kind == "expanded":
            present = bool(relevant) and max(calls.values(), default=0) >= expected_count
            criterion = f"MMA-containing expansion with at least {expected_count} call sites to one helper"
        else:
            present = len(relevant) >= expected_count and len(destinations) >= expected_count
            criterion = f"at least {expected_count} static {expected_kind} instructions and distinct destination register groups"
        report.update(status="static_evidence_present" if present else "inspect_manually",
                      criterion=criterion, disassembly_command=command, disassembly=str(path),
                      disassembly_sha256=sha256(path), static_math_instruction_counts={op: len(ops) for op, ops in math.items()},
                      static_instruction_operands=math, static_helper_call_sites=dict(calls),
                      relevant_destination_registers=destinations,
                      counting_caution="static helper bodies are shared across call sites; loop/control-flow inspection is necessary before a dynamic count")
        reports.append(report)
        print(f"sm_{arch} {label}: {report['status']} {report['static_math_instruction_counts']}")
    summary = {"cuobjdump_version": version, "results": reports,
               "limitation": "Bounded source-specific audit. This is neither a general SASS verifier nor an on-device validation result."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if all(report["status"] == "static_evidence_present" for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
