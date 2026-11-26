# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finite, inspectable coverage matrix. No claim that a finite suite exhausts CUDA."""
from __future__ import annotations
import itertools as it
from collections.abc import Iterator
from .model import Case

PROFILES = ("smoke", "standard", "exhaustive", "full")
INTEGER_TYPES = ("i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64")
FLOAT_TYPES = ("f16", "bf16", "f32", "f64")
CONVERT_TYPES = INTEGER_TYPES + FLOAT_TYPES
NARROW_TYPES = ("e2m1", "e2m3", "e3m2", "e4m3", "e5m2")
QUANT_FORMATS = tuple(f"{sign}{bits}" for sign in ("s", "u") for bits in range(1, 17)) + (
    "e2m1", "e2m3", "e3m2", "e4m3", "e5m2", "nf4", "f16", "bf16")
FAMILIES = ("integer", "integer_width", "scalar", "bit", "packed", "sfu", "convert", "tensor",
            "gemm", "blas", "quant", "quant_gemm", "memory", "atomic", "collective",
            "workload", "transfer", "launch", "sparse", "fft")

# Not counted as executed tests. Each has an integration point or a documented gap.
EXTENSIONS = [
    {"name": "CUTLASS profiler", "kind": "adapter", "coverage": "dense, sparse 2:4, mixed input, grouped, block scaled GEMM; installed kernel catalog is authoritative"},
    {"name": "Nsight Compute", "kind": "adapter", "coverage": "architecture-specific counters, roofline sections and SASS; --query-metrics determines availability"},
    {"name": "Compute Sanitizer", "kind": "adapter", "coverage": "memcheck, racecheck, initcheck, synccheck on selected reproducible cases"},
    {"name": "NCCL tests", "kind": "external_command", "coverage": "all-reduce, all-gather, reduce-scatter, broadcast, point-to-point multi-GPU collectives"},
    {"name": "FlashAttention forward/backward", "kind": "external_command", "coverage": "requires the actual implementation; bundled attention is a reference CUDA baseline"},
    {"name": "GGUF K/IQ, GPTQ, AWQ, EXL2/EXL3", "kind": "external_command", "coverage": "format-specific producer/layout/runtime required; never relabeled as generic int4"},
    {"name": "FP128 / arbitrary precision floating point", "kind": "gap", "coverage": "not implemented; no native CUDA scalar FP128 type assumed"},
    {"name": "TMA tensor maps / distributed shared memory", "kind": "gap", "coverage": "not directly benchmarked by core; CUTLASS kernels may exercise TMA, profiler evidence required"},
    {"name": "graphics / ray tracing / video / optical flow", "kind": "out_of_scope", "coverage": "not CUDA numeric compute benchmarks"},
]


def case(family: str, **params) -> Case:
    return Case(family, params)


def shapes(profile: str) -> list[tuple[int, int, int, str]]:
    base = [(16, 16, 16, "tiny"), (63, 65, 127, "ragged"), (256, 256, 256, "small"),
            (1024, 1024, 1024, "square"), (4096, 4096, 4096, "compute"),
            (1, 4096, 4096, "decode"), (16, 8192, 4096, "decode"),
            (128, 14336, 4096, "mlp"), (4096, 128, 4096, "skinny")]
    if profile == "smoke":
        return base[:2]
    if profile == "exhaustive":
        base += [(s, s, s, "square") for s in (32, 64, 128, 512, 2048, 8192, 16384)]
        base += [(m, n, k, "decode") for m, n, k in it.product((1, 2, 4, 8, 16, 32, 64), (4096, 8192, 16384), (4096, 8192))]
        base += [(m, n, k, "prefill") for m, n, k in it.product((128, 512, 2048), (4096, 14336, 28672), (4096, 7168))]
        base += [(m, n, k, "boundary") for m, n, k in ((31, 33, 65), (127, 129, 255), (255, 257, 513), (511, 513, 1025), (1000, 1001, 1003))]
    return base


def instruction_cases(profile: str) -> Iterator[Case]:
    modes = (("latency", 1), ("throughput", 4)) if profile != "exhaustive" else (("latency", 1),) + tuple(("throughput", x) for x in (1, 2, 4, 8))
    threads = (128,) if profile != "exhaustive" else (32, 64, 128, 256, 512)
    loops = (64,) if profile == "smoke" else ((256,) if profile == "standard" else (64, 256, 1024))
    int_ops = ("add", "sub", "mul", "mad", "div", "rem", "min", "max", "and", "or", "xor", "shl", "shr")
    float_ops = ("add", "sub", "mul", "fma", "div", "sqrt", "min", "max", "abs")
    for mode, ilp in modes:
        for t, iterations in it.product(threads, loops):
            controls = dict(mode=mode, ilp=ilp, threads=32 if mode == "latency" else t, iterations=iterations)
            if mode == "latency": controls["blocks"] = 1
            extra_widths = range(1, 129) if profile == "exhaustive" else ((1, 2, 3, 4, 5, 6, 7, 12, 24, 48, 96, 128) if profile == "standard" else (128,))
            int_types = INTEGER_TYPES + tuple(f"{sign}{bits}" for bits in extra_widths for sign in ("i", "u") if bits not in (8, 16, 32, 64))
            for dtype, op in it.product(int_types, int_ops if profile != "smoke" else ("add", "mad", "div")):
                yield case("integer", dtype=dtype, op=op, **controls)
            for dtype, op in it.product(FLOAT_TYPES, float_ops if profile != "smoke" else ("fma", "div")):
                yield case("scalar", dtype=dtype, op=op, **controls)
            for dtype in ("u32", "u64"):
                for op in ("popc", "clz", "brev") + (("prmt", "lop3", "shf") if dtype == "u32" else ()):
                    yield case("bit", dtype=dtype, op=op, **controls)
            for dtype, op in it.product(("f16x2", "bf16x2"), ("add", "mul", "fma")):
                yield case("packed", dtype=dtype, op=op, **controls)
            for a, b in it.product(("i8", "u8"), repeat=2):
                yield case("packed", dtype=a, op="dp4a", **{"dtype-b": b}, **controls)
            for a, b, op in it.product(("i16", "u16"), ("i8", "u8"), ("dp2a_lo", "dp2a_hi")):
                yield case("packed", dtype=a, op=op, **{"dtype-b": b}, **controls)
            for op, precision in it.product(("rcp", "rsqrt", "sin", "cos", "exp2", "log2"), ("approx", "precise")):
                yield case("sfu", dtype="f32", op=op, precision=precision, **controls)
    for src, dst in it.product(CONVERT_TYPES, CONVERT_TYPES):
        ignores_rounding = (src == dst or (src in INTEGER_TYPES and dst in INTEGER_TYPES) or
                            (dst == "f32" and src in ("f16", "bf16", "f32")) or
                            (dst == "f64" and src not in ("i64", "u64")))
        rounds = ("rn",) if profile != "exhaustive" or ignores_rounding else ("rn", "rz", "rd", "ru")
        for rounding in rounds:
            yield case("convert", dtype=src, **{"dtype-b": dst, "round": rounding}, op="cast", size=4099 if profile == "smoke" else 1048576)
    widths = (1, 4, 7, 24, 48, 64) if profile == "smoke" else (range(1, 65) if profile == "exhaustive" else (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64))
    for bits, signed, op in it.product(widths, (False, True), ("add", "sub", "mul", "and", "or", "xor", "shl", "shr", "min", "max")):
        yield case("integer_width", op=op, bits=bits, signed=signed, elements=4097 if profile == "smoke" else 1048576)


def tensor_cases(profile: str) -> Iterator[Case]:
    controls = (("latency", 32, 1), ("throughput", 128, 4)) if profile != "exhaustive" else (("latency", 32, 1),) + tuple(("throughput", t, ilp) for t, ilp in it.product((32, 64, 128, 256), (1, 2, 4, 8)))
    pairs = [(x, x) for x in ("f16", "bf16", "tf32", "f64", "i4", "u4", "b1_and", "b1_xor")]
    pairs += list(it.product(("i8", "u8"), repeat=2))
    pairs += [("i4", "u4"), ("u4", "i4")]
    pairs += list(it.product(NARROW_TYPES, repeat=2))
    for mode, t, ilp in controls:
        for a, b in pairs:
            scales = ("none", "mx") if a in NARROW_TYPES and b in NARROW_TYPES else ("none",)
            if a == b == "e2m1": scales += ("nv",)
            for scale in scales:
                accumulators = ("i32",) if a in ("i8", "u8", "i4", "u4", "b1_and", "b1_xor") else (("f64",) if a == "f64" else ("f32",))
                if scale == "none" and ((a == b == "f16") or (a in NARROW_TYPES and b in NARROW_TYPES)):
                    accumulators += ("f16",)
                for backend, accumulator in it.product(("native", "emulated"), accumulators):
                    patterns = ("ones", "alternating_sign") if profile == "exhaustive" and mode == "throughput" and a not in ("u8", "u4", "b1_and", "b1_xor") else ("ones",)
                    for pattern in patterns:
                        yield case("tensor", op="mma", dtype=a, **{"dtype-b": b}, scale=scale,
                                   backend=backend, accumulator=accumulator, pattern=pattern,
                                   threads=t, ilp=ilp, mode=mode, iterations=64 if profile == "smoke" else 256)
    if profile != "smoke":
        for a, b, scale, ea, eb in it.product(NARROW_TYPES, NARROW_TYPES, ("mx",), (-2, 0, 2), (-2, 0, 2)):
            if profile == "standard" and (a != b or (ea, eb) not in ((-2, 0), (0, 2))): continue
            yield case("tensor", op="mma", dtype=a, **{"dtype-b": b, "scale-exp-a": ea, "scale-exp-b": eb}, scale=scale,
                       backend="native", accumulator="f32", pattern="ones", threads=128, ilp=4, mode="throughput", iterations=256)

    for backend, mode in it.product(("native", "emulated"), ("latency", "throughput")):
        yield case("tensor", op="mma", dtype="e2m1", **{"dtype-b": "e2m1", "mma-kind": "expanded"}, scale="mx",
                   backend=backend, mode=mode, threads=32 if mode == "latency" else 128,
                   ilp=1 if mode == "latency" else 4, iterations=64 if profile == "smoke" else 256)


def gemm_cases(profile: str) -> Iterator[Case]:
    variants = [("f16", "f16", "f16", "default"), ("f16", "f16", "f32", "default"),
                ("bf16", "bf16", "bf16", "default"), ("bf16", "bf16", "f32", "default"),
                ("f32", "f32", "f32", "pedantic"), ("f32", "f32", "f32", "tf32"),
                ("f32", "f32", "f32", "fast_bf16"), ("f32", "f32", "f32", "fast_f16"),
                ("f32", "f32", "f32", "bf16x9"), ("f64", "f64", "f64", "default"),
                ("f64", "f64", "f64", "fixedpoint"), ("i8", "i8", "i32", "default"),
                ("c64", "c64", "c64", "default"), ("c128", "c128", "c128", "default")]
    narrow = [(a, b, d, "default") for a, b, d in it.product(("fp8_e4m3", "fp8_e5m2"), ("fp8_e4m3", "fp8_e5m2"), ("f16", "bf16", "f32"))]
    narrow += [("fp4_e2m1", "fp4_e2m1", d, "default") for d in ("f16", "bf16", "f32")]
    orientations = (("N", "N"), ("T", "N")) if profile != "exhaustive" else tuple(it.product(("N", "T"), repeat=2))
    batches = (1,) if profile != "exhaustive" else (1, 4, 16)
    for backend in ("cublas", "cublaslt"):
        for a, b, d, compute in variants + (narrow if backend == "cublaslt" else []):
            if backend == "cublaslt" and a in ("c64", "c128"): continue
            for m, n, k, regime in shapes(profile):
                for ta, tb in orientations:
                    if a.startswith("fp") and (ta, tb) != ("T", "N"): continue
                    for batch in batches:
                        pads = (0, 3) if profile == "exhaustive" else (0,)
                        for pad in pads:
                            params = {"backend": backend, "dtype": a, "dtype-b": b, "output-dtype": d,
                                      "compute": compute, "m": m, "n": n, "k": k, "trans-a": ta,
                                      "trans-b": tb, "batch": batch, "ld-pad": pad, "alpha": 1.0,
                                      "beta": 0.0, "workspace-mib": 64, "algorithms": 8, "epilogue": "default"}
                            if backend == "cublas": params.pop("algorithms")
                            if a == "fp4_e2m1": params["scale-mode"] = "block16"
                            yield Case("gemm", params, (regime,))
                            if a.startswith("fp8") and profile != "smoke":
                                yield Case("gemm", {**params, "scale-mode": "block32"}, (regime, "block_scaled"))
                            if backend == "cublas" and batch > 1 and profile == "exhaustive":
                                yield Case("gemm", {**params, "batch-mode": "pointer", "batch-gap": 256}, (regime, "pointer_batch"))
    if profile != "smoke":
        # A focused epilogue/alpha/beta/workspace grid, not a meaningless full Cartesian explosion.
        for dtype, epilogue, beta, workspace in it.product(("f16", "bf16", "f32"),
                ("default", "bias", "relu", "gelu", "bias_relu", "bias_gelu"),
                (0.0, 0.5), (0, 64, 256) if profile == "exhaustive" else (64,)):
            yield case("gemm", backend="cublaslt", dtype=dtype, m=257, n=511, k=1024,
                       alpha=0.75, beta=beta, epilogue=epilogue, **{"workspace-mib": workspace, "algorithms": 16})
    for dtype, op, n in it.product(("f32", "f64", "c64", "c128"),
             ("axpy", "dot", "nrm2", "scal", "asum", "iamax", "iamin", "copy", "swap", "gemv", "geam", "trsm", "trmm", "syrk", "symm", "hemm"),
             (127,) if profile == "smoke" else ((257, 1024) if profile == "standard" else (1, 31, 257, 1024, 4096))):
        if op == "hemm" and dtype not in ("c64", "c128"): continue
        if op in ("axpy", "dot", "nrm2", "scal", "asum", "iamax", "iamin", "copy", "swap"):
            yield case("blas", dtype=dtype, op=op, elements=n)
        else: yield case("blas", dtype=dtype, op=op, n=n, m=n, k=n)
    if profile != "smoke":
        for dtype, op, inc in it.product(("f32", "f64", "c64", "c128"), ("axpy", "dot", "nrm2", "scal", "asum", "iamax", "iamin", "copy", "swap"), (1, 2, 7)):
            yield case("blas", dtype=dtype, op=op, elements=65537, incx=inc, incy=inc)
        for dtype, side, uplo, diag in it.product(("f32", "f64", "c64", "c128"), ("left", "right"), ("lower", "upper"), ("unit", "nonunit")):
            yield case("blas", dtype=dtype, op="trsm", m=63, n=65, side=side, uplo=uplo, diag=diag)
        for strategy, control, bits in it.product(("eager", "performant"), ("dynamic", "fixed"), (0, 64, 128) if profile == "exhaustive" else (64,)):
            yield case("gemm", dtype="f64", compute="fixedpoint", backend="cublas", m=1024, n=1024, k=1024,
                       **{"emulation-strategy": strategy, "mantissa-control": control, "mantissa-bits": bits})


def quant_cases(profile: str) -> Iterator[Case]:
    sizes = (4099,) if profile == "smoke" else ((65537, 1048576) if profile == "standard" else (1, 31, 32, 33, 4099, 65537, 1048576, 16777216))
    groups = (32,) if profile == "smoke" else ((16, 32, 128) if profile == "standard" else (1, 16, 32, 64, 128, 256, 1024))
    formats = ("s1", "s4", "u4", "s8", "e2m1", "e2m3", "e3m2", "e4m3", "e5m2", "nf4", "f16", "bf16") if profile == "smoke" else QUANT_FORMATS
    for fmt, n, group, op in it.product(formats, sizes, groups, ("encode", "decode", "roundtrip")):
        yield case("quant", format=fmt, elements=n, group=group, op=op, **{"scale-format": "f32"})
    if profile != "smoke":
        for fmt, distribution, group in it.product(formats, ("zeros", "constant", "ramp", "outliers"), (16, 32, 128)):
            yield case("quant", format=fmt, elements=4099, group=group, op="roundtrip", distribution=distribution, **{"scale-format": "f32"})
    # Format encoding and scale format are independent. These emulate block-scaled numeric values.
    for fmt, scale, group in ((f, "e8m0", 32) for f in NARROW_TYPES):
        yield case("quant", format=fmt, elements=65539, group=group, op="roundtrip", **{"scale-format": scale})
    yield case("quant", format="e2m1", elements=65539, group=16, op="roundtrip", **{"scale-format": "e4m3"})
    gm_shapes = [(4, 128, 256)] if profile == "smoke" else ([(1, 4096, 4096), (16, 1024, 1024), (128, 512, 1024)] if profile == "standard" else [(1, 4096, 4096), (8, 4096, 4096), (32, 4096, 4096), (128, 1024, 4096), (512, 1024, 4096), (17, 131, 257)])
    for fmt, a, shape, mode, group in it.product(formats, ("f16", "bf16", "f32") if profile != "smoke" else ("f32",), gm_shapes,
                                               ("fused", "dequant_gemm", "predequant_gemm", "end_to_end"), (32, 128) if profile == "exhaustive" else (32,)):
        m, n, k = shape
        yield case("quant_gemm", format=fmt, dtype=a, m=m, n=n, k=k, group=group, mode=mode)


def system_cases(profile: str) -> Iterator[Case]:
    sizes = (4096,) if profile == "smoke" else ((4096, 262144, 16777216) if profile == "standard" else (32, 1024, 8192, 65536, 262144, 1048576, 16777216, 67108864, 268435456))
    for n in sizes:
        for op in ("read", "write", "copy", "triad", "strided", "gather", "scatter", "cp_async", "shared_copy"):
            strides = (1, 2, 4, 8, 16, 32, 33) if op == "strided" and profile == "exhaustive" else ((1, 32) if op == "strided" else (1,))
            for stride in strides:
                params = {"op": op, "dtype": "f32", "elements": n}
                if op == "strided": params["stride"] = stride
                yield Case("memory", params)
        # Full cycles for small working sets, bounded random prefixes for large ones.
        # The case records the exact number of dependent loads; never calls N a cache footprint.
        yield case("memory", op="pointer_chase", dtype="u32", elements=n, iterations=min(max(n, 4096), 1048576))
        for op in ("shuffle", "ballot", "reduce_sum", "block_reduce", "block_scan", "scan", "sort"):
            yield case("collective", op=op, elements=n, dtype="u32" if op == "ballot" else "f32")
    shared_strides = (1, 32) if profile != "exhaustive" else (1, 2, 4, 8, 16, 31, 32)
    block_sizes = (128,) if profile != "exhaustive" else (32, 64, 128, 256, 512)
    for stride, threads in it.product(shared_strides, block_sizes):
        yield case("memory", op="shared_bank", dtype="f32", stride=stride, threads=threads, iterations=256)
    for threads in block_sizes:
        yield case("collective", op="barrier", dtype="u32", threads=threads, iterations=256)
    for dtype in ("i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f32", "f64", "f16", "bf16"):
        ops = ("add",) if dtype in ("f16", "bf16") else (("add", "cas", "min", "max", "exchange", "and", "or", "xor") if dtype[0] in "iu" else ("add", "min", "max", "cas", "exchange"))
        for op, addresses in it.product(ops, (1, 32, 4096) if profile != "exhaustive" else (1, 2, 8, 32, 128, 4096, 65536)):
            yield case("atomic", dtype=dtype, op=op, elements=max(addresses, 4096 if profile == "smoke" else 65536), addresses=addresses, iterations=1)
    workload_types = ("f32",) if profile == "smoke" else FLOAT_TYPES
    for op, rows, cols, dtype in it.product(("softmax", "rmsnorm", "layernorm", "rope"),
            (4,) if profile == "smoke" else (1, 16, 128),
            (127,) if profile == "smoke" else ((128, 1024, 4096) if profile == "standard" else (31, 127, 128, 1024, 4096, 8192, 16384)), workload_types):
        yield case("workload", op=op, dtype=dtype, rows=rows, cols=cols + (cols % 2) if op == "rope" else cols)
    activation_sizes = (4099,) if profile == "smoke" else ((128, 1024, 16384, 131072, 1048576) if profile == "standard" else (1, 31, 4099, 65536, 1048576, 16777216))
    for op, n, dtype in it.product(("relu", "gelu", "gelu_tanh", "silu", "swiglu"), activation_sizes, workload_types):
        yield case("workload", op=op, dtype=dtype, elements=n)
    for sequence, head_dim, causal, dtype in it.product((16,) if profile == "smoke" else (64, 256, 1024), (64, 128), (False, True), workload_types):
        yield case("workload", op="attention", dtype=dtype, sequence=sequence, **{"head-dim": head_dim}, causal=causal)
    if profile == "exhaustive":
        for dtype, heads, batch in it.product(FLOAT_TYPES, (1, 8, 32), (1, 4)):
            yield case("workload", op="attention", dtype=dtype, sequence=128, **{"head-dim": 64}, heads=heads, batch=batch, causal=True)
    yield case("launch", op="empty", iterations=100)
    for nodes in ((16,) if profile != "exhaustive" else (1, 16, 128)):
        yield case("launch", op="graph", iterations=100, **{"graph-nodes": nodes})
    for op, n in it.product(("h2d", "d2h", "d2d", "managed_prefetch", "managed_first_touch", "p2p"), sizes):
        for pinned in ((False, True) if op in ("h2d", "d2h") else (True,)):
            params = {"op": op, "elements": n}
            if op in ("h2d", "d2h"): params["pinned"] = pinned
            yield Case("transfer", params)
    for n in ((256,) if profile == "smoke" else (1024, 4096, 65536)):
        for dtype, op, nnz in it.product(("f32", "f64"), ("csr_spmv", "csr_spmm"), (1, 8, 32) if profile == "exhaustive" else (8,)):
            params = {"op": op, "dtype": dtype, "rows": n, "cols": n, "nnz-per-row": nnz}
            if op == "csr_spmm": params["n"] = 16
            yield Case("sparse", params)
        for dtype, op in it.product(("f16", "bf16", "f32", "f64"), ("c2c", "r2c", "c2r")):
            directions = ("forward", "inverse") if op == "c2c" and profile != "smoke" else (("inverse",) if op == "c2r" else ("forward",))
            for direction in directions: yield case("fft", op=op, dtype=dtype, n=n, direction=direction)
    if profile != "smoke":
        for dtype, rank, op, batch in it.product(("f16", "bf16", "f32", "f64"), (2, 3), ("c2c", "r2c", "c2r"), (1, 4)):
            params = {"op": op, "dtype": dtype, "rank": rank, "nx": 32, "ny": 16, "batch": batch, "direction": "inverse" if op == "c2r" else "forward"}
            if rank == 3: params["nz"] = 8
            yield Case("fft", params)
    if profile == "exhaustive":
        for dtype, op, n in it.product(("f32", "f64"), ("c2c", "r2c", "c2r"), (127, 1000, 4095, 10000)):
            yield case("fft", op=op, dtype=dtype, n=n, direction="inverse" if op == "c2r" else "forward")


def generate(profile: str = "standard") -> Iterator[Case]:
    if profile not in PROFILES: raise ValueError(f"unknown profile {profile}")
    seen: set[str] = set()
    if profile == "full":
        from pathlib import Path
        from .config import load_cases
        sources = [generate(name) for name in ("smoke", "standard", "exhaustive")]
        sources += [load_cases(path) for path in sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.json"))]
        for source in sources:
            for item in source:
                if item.id not in seen:
                    seen.add(item.id)
                    yield item
        return
    for generator in (instruction_cases, tensor_cases, gemm_cases, quant_cases, system_cases):
        for item in generator(profile):
            if item.id not in seen:
                seen.add(item.id)
                yield item


def select(cases: Iterator[Case], *, families=(), dtypes=(), ops=(), match="", shard=None, limit=0) -> Iterator[Case]:
    emitted = 0
    for item in cases:
        if families and item.family not in families: continue
        if dtypes and item.params.get("dtype", item.params.get("format")) not in dtypes: continue
        if ops and item.params.get("op", "gemm" if item.family.endswith("gemm") else "") not in ops: continue
        if match and match not in item.label: continue
        if shard is not None:
            index, count = shard
            if count < 1 or index < 0 or index >= count: raise ValueError("shard must be INDEX/COUNT, zero based")
            if int(item.id.split(".")[-1], 16) % count != index: continue
        yield item
        emitted += 1
        if limit and emitted >= limit: return
