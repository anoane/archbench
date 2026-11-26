# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compile-only capability evidence. These never count as measured/passed GPU tests."""
from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
from .runner import atomic_json, utc_now

PTX_PROBES = {
    "fma_f32": (".reg .f32 a,b,c,d; mov.f32 a,1.0; mov.f32 b,1.0; mov.f32 c,0.0; fma.rn.f32 d,a,b,c;", "st.global.f32 [p],d;"),
    "fma_f64": (".reg .f64 a,b,c,d; mov.f64 a,0d3ff0000000000000; mov.f64 b,a; mov.f64 c,0d0000000000000000; fma.rn.f64 d,a,b,c;", "st.global.f64 [p],d;"),
    "fma_bf16": (".reg .b16 a,b,c,d; mov.b16 a,0x3f80; mov.b16 b,a; mov.b16 c,0; fma.rn.bf16 d,a,b,c;", "st.global.b16 [p],d;"),
    "add_bf16": (".reg .b16 a,b,d; mov.b16 a,0x3f80; mov.b16 b,a; add.rn.bf16 d,a,b;", "st.global.b16 [p],d;"),
    "dp4a_signed_unsigned": (".reg .b32 a,b,d; mov.b32 a,0x01010101; mov.b32 b,0x02020202; dp4a.s32.u32 d,a,b,0;", "st.global.b32 [p],d;"),
    "mma_f64_m8n8k4": (".reg .f64 a,b,d0,d1; mov.f64 a,0d3ff0000000000000; mov.f64 b,a; mov.f64 d0,0d0000000000000000; mov.f64 d1,d0; mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {d0,d1},{a},{b},{d0,d1};", "st.global.f64 [p],d0;"),
    "mma_f16_m16n8k16": (".reg .b32 a<4>,b<2>; .reg .f32 d<4>; mov.b32 a0,0x3c003c00; mov.b32 a1,a0; mov.b32 a2,a0; mov.b32 a3,a0; mov.b32 b0,a0; mov.b32 b1,a0; mov.f32 d0,0.0; mov.f32 d1,0.0; mov.f32 d2,0.0; mov.f32 d3,0.0; mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {d0,d1,d2,d3},{a0,a1,a2,a3},{b0,b1},{d0,d1,d2,d3};", "st.global.f32 [p],d0;"),
}


def compile_probes(output: Path, architectures: list[str], ptxas=None):
    compiler = ptxas or shutil.which("ptxas")
    if not compiler: raise ValueError("ptxas missing; compile probes require the CUDA toolkit")
    if any(x not in ("80", "89", "120a") for x in architectures): raise ValueError("unsupported target")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for arch in architectures:
        for name, (body, store) in PTX_PROBES.items():
            stem = f"{name}.sm_{arch}"
            source = output / (stem + ".ptx")
            source.write_text(f".version 9.5\n.target sm_{arch}\n.address_size 64\n.visible .entry probe(.param .u64 output) {{\n.reg .b64 p; ld.param.u64 p,[output];\n{body}\n{store}\nret;\n}}\n")
            command = [compiler, "-arch=sm_" + arch, str(source), "-o", str(output / (stem + ".cubin")), "-v"]
            process = subprocess.run(command, capture_output=True, text=True, timeout=90)
            (output / (stem + ".log")).write_text(process.stdout + process.stderr)
            results.append({"feature": name, "architecture": arch, "status": "compiled" if process.returncode == 0 else "compile_rejected",
                            "argv": command, "returncode": process.returncode, "gpu_execution_verified": False,
                            "meaning": "assembler acceptance only; rejection may indicate toolkit version, syntax, or target restrictions"})
    report = {"schema_version": 1, "created_at": utc_now(), "results": results, "counts_as_benchmark_coverage": False}
    atomic_json(output / "capabilities.json", report)
    return report
