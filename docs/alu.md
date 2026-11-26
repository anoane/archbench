# Arithmetic, conversion, and warp-MMA benchmarks

`src/alu.cu` implements numerical benchmarks, not a list of placeholder cases.
Every measured case validates output values against a host reference. It returns
`validation_failed` for a mismatch and reports unsupported combinations
explicitly. A successful host test is not evidence that CUDA compilation or GPU
execution succeeded. Those remain separate gates in this delivery.

## Implemented families

| Family | Types | Operations / controls |
| --- | --- | --- |
| `integer` | `i8/u8`, `i16/u16`, `i32/u32`, `i64/u64` | `add`, `sub`, `mul`, `mad`, `div`, `rem`, `min`, `max`, `and`, `or`, `xor`, `shl`, `shr` |
| `integer` | Every other `iN/uN` width, N=1..128 | Same operations, software arithmetic on two 64-bit limbs |
| `scalar` | `f16`, `bf16`, `f32`, `f64` | `add`, `sub`, `mul`, `fma`, `div`, `sqrt`, `min`, `max`, `abs` |
| `packed` | `f16x2`, `bf16x2` | `add`, `mul`, `fma`, two independent values per packed register |
| `packed` | A=`i8/u8`, B=`i8/u8` | `dp4a`, all four signedness combinations |
| `packed` | A=`i16/u16`, B=`i8/u8` | `dp2a_lo`, `dp2a_hi`, all four signedness combinations |
| `bit` | `u32`, `u64` | `popc`, `clz`, `brev` |
| `bit` | `u32` | `prmt`, `lop3`, `shf` |
| `sfu` | `f32` | `rcp`, `rsqrt`, `sin`, `cos`, `exp2`, `log2`; `precision=approx/precise` |
| `convert` | All ordered pairs from `{i8,u8,i16,u16,i32,u32,i64,u64,f16,bf16,f32,f64}` | Typed streaming casts, `round=rn/rz/rd/ru` |
| `tensor` | See the matrix below | Register-resident dense warp `mma`, dependent or independent accumulator chains |

The arithmetic kernels use independent integer/float recurrences. Integer and
bitwise outputs are compared exactly. Floating recurrence comparisons use
explicit tolerances. Conversion streams check every output element. Tensor
microbenchmarks check every output register using exactly representable,
uniform matrix fragments. These checks establish the advertised baseline's
semantics; they do not replace random-matrix GEMM validation or exhaustive
floating-point exception testing.

Sampled recurrence checks select at most 256 values using stratified thread
lanes and then include every ILP chain and every packed component of each
selected lane. They do not stride through a flattened register array in a way
that can repeatedly select chain zero. Scalar, packed-float, and SFU paths also
scan every downloaded output for nonfinite values. Integer failure diagnostics
subtract unsigned encoded bit patterns before conversion to double, preserving
one-bit/unit differences even near the largest 64/128-bit values. Exact equality
remains the integer acceptance gate; the reported distance is explicitly a
bit-pattern metric. Half-accumulator tensor failures decode both components
to report numeric error, including infinite metrics for NaN/Inf mismatches.

## Timing and dependency controls

`--mode latency` requires exactly one block, one warp, and `--ilp 1`.
`--mode throughput` permits `--ilp 1`, `2`, `4`, or `8`, multiple blocks, and
multiple warps. Threads must be a multiple of 32. Common controls are
`--iterations`, `--threads`, `--blocks`, and the shared event-timing options.
`--blocks-per-sm 1..64` chooses a grid proportional to the selected device's
SM count. It is exclusive with `--blocks` and is unavailable in latency mode.
Compare work-normalized rates for grids proportional to SM count: equal case
parameters can execute different total work on devices with different SM counts.

Each launch initializes its own accumulators. Repeating a launch therefore does
not silently accumulate state across warmup, calibration, and timing samples.
The kernel stores every final chain value. Explicit PTX is volatile; CUDA
intrinsics remain subject to compiler lowering and should be checked in SASS.

The primary timings are CUDA events through the shared harness. Each arithmetic
kernel also records a `clock64()` interval. The metric
`minimum_clock_cycles_per_step` is the minimum observed per-thread interval
divided by the iteration count **in the last measured launch**. It includes
loop control, register dependencies, and warp scheduling. It is not a
single-instruction SASS latency measurement. In throughput mode it is not
divided by ILP and should not be used as a substitute for aggregate throughput.
Minimum, median, and maximum per-thread clock intervals are also recorded for
the last measured launch. They are raw cycle intervals, not times inferred from
a nominal clock rate.

The SFU recurrence applies `fma(x,0.125,1.5)` before each evaluation to keep
arguments bounded and valid. Reported useful work counts one SFU evaluation;
the maintenance FMA and loop overhead remain in the measured time. `precise`
uses CUDA mathematical functions or explicit round-to-nearest operations;
`approx` requests the corresponding approximate PTX instruction. Neither label
means correctly rounded transcendental functions in every input domain.

Conversion cases are memory streams, not register-latency tests. `--size` is
the number of elements. Reads and writes are included in timing. Rounding is
irrelevant for integral-to-integral casts. Identity conversions are labeled
`identity_copy`. Low floating destinations currently use an explicit FP32
intermediate, including FP64→FP16/BF16; that path is documented in each result.
Input generation stays inside destination integer ranges, avoiding undefined
out-of-range C++ conversion behavior. These throughput cases do not claim to
exercise every conversion edge value.

## Integer semantics

The native-width routes request PTX. Eight- and sixteen-bit values are promoted
to 32-bit operations and wrapped to their declared width after each step. They
are not mislabeled as dedicated scalar 8-bit/16-bit hardware ALUs. Sixty-four-bit
PTX often lowers into several machine instructions; disassembly is required
before making claims about native SASS execution.

Add, subtract, multiply, and multiply-add are explicitly modulo 2^N. Signed
comparisons interpret two's-complement bit patterns. Signed division truncates
toward zero. The 8/16/32/64 paths require a positive divisor after conversion,
which excludes division by zero and the signed `MIN/-1` overflow case. Their
shift count is masked to the width and signed right shifts sign-extend.

The nonstandard-width implementation stores each number in two `uint64_t`
limbs. Add/subtract propagate carries, multiplication computes a truncated
128-bit product, and division performs binary long division. Signed division
uses unsigned magnitudes. Every result is masked to the requested width.
`MIN/-1` is defined to wrap modulo 2^N. The shift count is the low 64 bits of
the operand modulo the width. This makes even signed one-bit arithmetic
well-defined. It is an emulation baseline, not a claim of optimal sub-byte code.

Run the host arithmetic audit with:

```bash
python3 tools/check_alu_reference.py --compiler g++
```

The script extracts the actual two-limb algorithms, replaces only the CUDA
high-multiply intrinsic with an equivalent host expression, and compares them
with an independent native `unsigned __int128` reference. It checks 532,480
transitions across widths 1..128, both signedness choices, 13 operations,
seeded random values, zeros, sign boundaries, full-width carries, and division
overflow boundaries. It does not validate PTX or GPU hardware. Linux/WSL2 with
a host compiler supporting `__int128` is the intended platform.

## Dense tensor microbenchmark matrix

`--family tensor --op mma` accepts `--dtype` for A, `--dtype-b` for B,
`--accumulator`, `--scale`, and `--backend native|emulated`.

| A × B | Shape | Accumulator | Native worker policy |
| --- | --- | --- | --- |
| FP16 × FP16 | 16×8×16 | FP32 or FP16 | SM80, SM89, SM120a |
| BF16 × BF16 | 16×8×16 | FP32 | SM80, SM89, SM120a |
| TF32 × TF32 | 16×8×8 | FP32 | SM80, SM89, SM120a |
| FP64 × FP64 | 8×8×4 | FP64 | Enabled only when the selected target's compiler probe accepts the exact instruction |
| `{i8,u8}` × `{i8,u8}` | 16×8×32 | INT32 | All three workers |
| `{i4,u4}` × `{i4,u4}` | 16×8×64 | INT32 | SM80/SM89; SM120a requires the exact target compiler probe to pass |
| Binary AND+POPC / XOR+POPC | 16×8×256 | INT32 | SM80/SM89; SM120a requires its separate target compiler probe to pass |
| `{e4m3,e5m2}` × `{e4m3,e5m2}` | 16×8×32 | FP32 or FP16 | SM89/SM120a |
| Any of `{e2m1,e2m3,e3m2,e4m3,e5m2}` × any of the same five | 16×8×32 | FP32 or FP16 | SM120a-specific narrow instructions |
| All 25 narrow pairs, MX scales | 16×8×32 | FP32 | SM120a; see FP4 exception below |
| E2M1 × E2M1, MXFP4 | 16×8×64 | FP32 | SM120a packed `mxf4` |
| E2M1 × E2M1, NVFP4 | 16×8×64 | FP32 | SM120a packed `mxf4nvf4`, `scale_vec::4X` |

FP8 with an FP16 accumulator uses the plain FP8 MMA instruction on SM89 and
SM120a. The current PTX instruction syntax and target requirements are documented
in NVIDIA's [MMA instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).
FP64 and SM120 subbyte paths use configure-time compile probes under
`cmake/probes/`. A rejected probe leaves that native feature unavailable while
its emulated counterpart remains callable. Probe acceptance does not prove
runtime correctness or a Tensor Core SASS implementation. Actual outputs must
pass validation, and hardware instruction claims still require disassembly or
profiler evidence. Results use `requested_tensor_ptx` for the native request,
or `compiler_expanded_tensor_ptx` when the exact-instruction probe identifies
an expanded implementation. `tensor_codegen_evidence` records the classification
and its probe-only evidence scope. A DMMA opcode in compiled code does not prove
that a particular physical GPU successfully executes that request.

The CUDA 13.5.37 compile probes inspected during this revision emit DMMA for
the FP64 instruction on all three targets. On SM120a, the legacy INT4 and binary
PTX probes expand into INT8 IMMA instructions plus conversion/control logic.
Those paths are not described as direct native INT4/BMMA execution. Preserve
the build's actual probe disassembly and classification when comparing results;
another compiler may lower the same PTX differently.

Narrow type names are precise: FP4=`e2m1`; FP6=`e2m3` or `e3m2`;
FP8=`e4m3` or `e5m2`. There is no four-bit E4M3 type.

For `scale=none` and expanded `scale=mx` instructions, FP4 occupies bits 5:2 of
each byte and FP6 occupies the low six bits. Packed MXFP4/NVFP4 uses contiguous
four-bit elements. MX scales use E8M0 and groups of 32. NVFP4 uses E4M3 and
groups of 16. For E2M1×E2M1, `--scale mx` defaults to packed MXFP4; add
`--mma-kind expanded` to measure the 16×8×32 expanded path instead.

`--scale-exp-a` and `--scale-exp-b` select exact power-of-two scale factors in
the exponent range -6..6. Default zero means scale=1. Every lane is given the
same scale metadata, so selectors are valid without depending on a complex
layout. This verifies scale application for the uniform baseline. It is not a
test of every scale-selector layout or the NVFP4 external global scale.

The default input pattern is `ones`. `--pattern alternating_sign` changes A
to -1 in alternating warps, for formats with a signed A representation. Every
accumulator register is checked. FP16 accumulator tests check exact final bit
patterns, including round-to-nearest behavior at each MMA boundary. A
nonfinite FP16 host recurrence is rejected as `not_validated` before timing,
with `accumulator_finite=false`. Finite recurrences that stop changing due to
rounding are labeled `accumulator_stagnated=true`.

`--reset-every N` divides a tensor launch into segments of at most N MMA
iterations, clearing the accumulators between segments. Zero disables resets.
The host reference checks the final segment, including a partial final segment.
Every requested MMA iteration still contributes to the useful-work count.
Clears and segment-loop overhead remain inside timing and are explicitly excluded
from mathematical work. This control permits long, finite FP16 runs; use
`--reset-every 512` for the bundled 20,000-iteration FP16-accumulator probes.
It also permits a long FP32 scalar-emulated run without exceeding the exact
2^24-addition limit within a segment. It does not eliminate loop overhead.

Useful work is `warps × ILP × iterations × 2MNK`. FMA counts as two operations.
Binary MMA uses `MNK` bit pairs and a separate `Gbit-pair/s` unit, so binary
results are not silently treated as floating-point FLOPs. Kernel timing excludes
matrix loads, quantization, packing, scale computation, and dequantization.
Exact integer work and tile counts are retained in metadata, in addition to
the floating-point rate numerator. Logical tile counts are not measured SASS
instruction counters. Floating scalar/packed arithmetic uses `Gflop/s`, integer
arithmetic uses `Gop/s`, SFU evaluations use `Gcall/s`, and conversions use
`Gconversion/s`; every family sets an explicit work domain.

The emulated backend computes the same **uniform decoded fragments** with
scalar arithmetic. It excludes packing/dequantization and is not an optimized
low-precision GEMM. FP16 accumulation emulation uses FP32 scalar dot products
and rounds to FP16 at each MMA boundary. FP32 emulation rejects more than 2^24
unit additions per segment so its exact reference cannot silently exceed
the representable integer range. Use the separate GEMM/quantization families
for end-to-end bandwidth and dequantization comparisons.

## Comparison with the linked GPU Unlock FP8 probe

The linked [FP8 source](https://github.com/JustAResearcher/gpu-unlock-ada/blob/main/docs/evidence/rtx4090-48g-modded/fp8-mma-bench.cu)
was retrieved and inspected on 2026-11-25. `configs/tensor_throughput.json`
contains 140 cases: a native/emulated pair matching its FP8 launch controls and
69 further native/emulated pairs for grid, data-type, accumulator, and scale
coverage. The first pair uses E4M3×E4M3→FP32, `m16n8k32`, 1,056 blocks,
256 threads per block, eight independent chains, and 20,000 iterations. Its
mathematical numerator is exactly **11,072,962,560,000 floating-point operations
per launch**. The wider sweep uses four, eight, or sixteen blocks per SM.

The upstream source times five individual launches with no warmup and reports
the fastest. The matrix supplies kernel controls only; the runner supplies
sample, warmup, and repeat policy. For a direct comparison choose five samples,
zero warmups, and one measured launch per sample. For a steadier comparison use
warmup and report the distribution as well as the best observation. The best
observation is not a guaranteed sustained peak or a chip-specification value.

This suite's implementation is independently written and keeps/checks every
final accumulator, while the upstream timed kernel writes one combined sink.
This suite also records per-thread `clock64` intervals. Those differences and
the new segment-loop control remain in timed work, so identical instructions
and launch controls do not imply identical binaries or overhead. No upstream
source code is redistributed with the suite.

The linked [GEMM source](https://github.com/JustAResearcher/gpu-unlock-ada/blob/main/docs/evidence/rtx4090-pr1-stock/gemm_bench.cu)
is a different workload: cuBLAS FP16/BF16 inputs with FP32 outputs, square sizes
1,024/4,096/8,192, ten warmups, and five batches of twenty calls. Its matrix
traffic and library algorithm selection must be compared with the GEMM family,
not used interchangeably with a register-resident MMA issue-rate result.

Run `python3 -m unittest discover -s tests -p test_tensor_model.py` with the
project's Python module directory on `PYTHONPATH` to check the exact counter
formula, segment tails, overflow guards, and paired throughput-plan controls.
These are host checks and provide no CUDA compilation or GPU execution evidence.

## Architecture-specific build audit

The project builds separate workers. `archbench_sm120a` must be compiled with
both `-gencode=arch=compute_120a,code=sm_120a` and
`ARCHBENCH_ARCH_SUFFIX_A=1`. Only that device compilation includes the narrow
PTX. The SM80 and SM89 device passes see guarded no-op definitions but cannot
legitimately execute them: runtime capability gates reject those native cases.
If a gate were bypassed, output validation would fail rather than publish a
successful zero-time tensor result.

FP64 tensor code requires `ARCHBENCH_HAS_FP64_MMA`, set only after the selected
target's compiler probe succeeds. Native FP8 with either FP32 or FP16
accumulators is compiled for architectures at least 890. INT4 and binary code
is present below 1200; the SM120 build independently requires
`ARCHBENCH_SM120_INT4_MMA` and `ARCHBENCH_SM120_BINARY_MMA`. These macros are set
by separate exact-instruction compiler probes. Standard FP16/BF16/TF32 and INT8
instructions remain available across the three requested compile targets.
SASS inspection is still required: PTX is a virtual instruction set.

`tools/audit_alu_sass.py` reproducibly checks a bounded set of compiled ILP=8
worker kernels. It saves exact extraction commands, object hashes, disassembly,
destination-register evidence, and helper-call counts. In the inspected CUDA
13.5.37 build, SM80 FP16→FP32, SM89 FP8→FP32/FP16, and SM120a FP4→FP32 retain
eight distinct MMA accumulator groups. FP32 scalar and FP16 packed FMA retain
eight arithmetic chains. SM120a INT4/binary expansions use shared out-of-line
helper bodies called eight times in the loop; counting only static IMMA lines
would undercount the requested work. These observations are compiled-code
evidence, with GPU execution still checked independently.

```bash
python3 tools/audit_alu_sass.py --build build --cuobjdump /usr/local/cuda-13.5/bin/cuobjdump
PYTHONPATH=python python3 -m unittest discover -s tests -p test_alu_validation.py -v
PYTHONPATH=python python3 -m unittest discover -s tests -p test_tensor_model.py -v
```

The validation-helper test covers 50 lane/ILP/packing shapes, nonfinite values
outside the recurrence samples, finite mismatch metrics, and adjacent encoded
64/128-bit integers. The accounting tests check the linked FP8 benchmark's
exact numerator, all 140 throughput-plan cases, and 54 reset schedules. These
host checks exercise actual portable helpers and host validation statements;
they do not replace compiler and hardware acceptance tests.

Two details in the supplied comparison require particular care. BF16
`add/sub/mul` PTX requires SM90, while BF16 FMA is available from SM80;
CUDA headers synthesize the former operations on older targets. Also, the
SM80 FP64 MMA shape is 8×8×4; the larger 16×8 FP64 shapes require SM90.
Scalar FP64 arithmetic remains distinct from FP64 Tensor Core support.

## On-device canaries before a long run

Build each target with the installed CUDA 13.5 compiler first. Preserve the
complete compiler log, PTX probe output, and disassembly. A successful build
establishes compiler acceptance only. Then run small cases on the matching GPU
before starting a standard or exhaustive campaign.

```bash
# Dependent FP32 FMA chain on the SM80 GPU.
build/bin/archbench_sm80 --device 0 --family scalar --dtype f32 --op fma \
  --mode latency --threads 32 --blocks 1 --ilp 1 --iterations 256

# Signed/unsigned packed dot product, with negative signed-byte test inputs.
build/bin/archbench_sm80 --device 0 --family packed --dtype i8 --dtype-b u8 \
  --op dp4a --iterations 128 --blocks 2 --threads 64 --ilp 2

# Complete 127-bit two-limb arithmetic and independent host reference.
build/bin/archbench_sm80 --device 0 --family integer --dtype i127 --op div \
  --iterations 8 --blocks 1 --threads 32 --ilp 1

# Uniform signed fragments verify a native narrow mixed-type path.
build/bin/archbench_sm120a --device 0 --family tensor --op mma \
  --dtype e3m2 --dtype-b e2m1 --backend native --scale none \
  --pattern alternating_sign --blocks 2 --threads 32 --ilp 1 --iterations 16

# Verify both scale factors are applied; their product here is 1/2.
build/bin/archbench_sm120a --device 0 --family tensor --op mma \
  --dtype e2m1 --dtype-b e2m1 --backend native --scale nv \
  --scale-exp-a -2 --scale-exp-b 1 --blocks 1 --threads 32 --ilp 1 --iterations 16

# FP16 accumulator versus the FP32 accumulator is a separate comparison.
build/bin/archbench_sm89 --device 0 --family tensor --op mma \
  --dtype f16 --dtype-b f16 --accumulator f16 --backend native \
  --blocks 1 --threads 32 --ilp 1 --iterations 32
```

Require `status=ok` and `validation=passed` from each applicable case. An
unsupported record is not a passing correctness check. Repeat supported
canaries under Compute Sanitizer memcheck, then inspect SASS to confirm the
requested operations and absence of unintended spills. Increase iteration
count and ILP only after these gates. Run the same logical case, timing policy,
and validation policy on each GPU; keep product-specific clocks, power limits,
memory systems, and active SM counts in the comparison metadata.

## Primary references

- [NVIDIA PTX ISA 9.5](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html): integer arithmetic, half/BF16 arithmetic, conversion rules, dense warp MMA, block scaling, and target ISA notes.
- [NVIDIA CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html): half/BF16 intrinsics and rounding modes.
- [NVIDIA CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html): SM120 narrow mixed-input and block-scaled MMA families.

These references describe available mechanisms. Measured support, correctness,
throughput, and SASS evidence must come from running the delivered suite on the
installed toolkit and each actual GPU.
