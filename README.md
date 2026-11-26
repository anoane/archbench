# ArchBench 0.2.0 — CUDA 13.5 architecture comparison

ArchBench compares **SM80, SM89 and SM120a** using separate exact-target workers,
a reproducible Python runner, numerical references, explicit software
counterparts, and reports that preserve failures and unavailable requests.
**FP32 and FP64 are included**, alongside integer, low-precision, mixed,
quantized, real and complex paths.

The `full` selection contains **404,129 distinct requested cases across 20
families**, including the union of every bundled profile/configuration and its
implemented software counterparts. This is a finite, inspectable catalogue,
not a claim to cover every CUDA program, shape, input bit pattern, proprietary
library kernel, or model quantization ecosystem. A complete run means every
selected case has a recorded outcome; it does not mean every request passed.

**Validation:** real CUDA **13.5.37** compilation and linking have been performed
for all three worker targets, with cuBLAS, cuBLASLt, cuFFT and cuSPARSE enabled.
Host arithmetic, orchestration, report, sampling and provenance checks are
included. This workspace has no NVIDIA GPU: the new kernels still need runtime
acceptance and Compute Sanitizer checks on your hardware. The four supplied
older runs are preserved as historical evidence, not relabeled as new results.
See [validation evidence](docs/validation.md) and [changes and findings](docs/release_0_2_0.md).

## Complete benchmark and test catalogue

The generated table below is the operation-level inventory of the effective
`full` selection: the deduplicated union of `smoke`, `standard`, `exhaustive`
and every bundled configuration, followed by default software-counterpart
expansion. It lists all 20 registered families, every operation name, every
type/format class, every requested backend, and the named semantic policies.
The shape/control tables then enumerate their numeric grids. It does not pretend
that a compact Markdown table can print 404,129 full case labels. The exact
JSONL export immediately below is the authoritative one-object-per-case list.

<!-- BEGIN GENERATED BENCHMARK CATALOG -->

This table is generated from the effective `full` selection after default
software-pair expansion. Contiguous integer and quantization widths are
compressed as inclusive ranges; every other operation, type, backend and
mode/policy value is written explicitly.

| Family | Cases | Operations | Types and formats | Backends | Modes and policies |
|---|---:|---|---|---|---|
| `integer` | 209,700 | `add`, `and`, `div`, `mad`, `max`, `min`, `mul`, `or`, `rem`, `shl`, `shr`, `sub`, `xor` | dtype: `i1`–`i128`, `u1`–`u128` | `core` | mode: `latency`, `throughput` |
| `integer_width` | 1,401 | `add`, `and`, `max`, `min`, `mul`, `or`, `shl`, `shr`, `sub`, `xor` | bits: `1`–`64`; <br>signed: `false`, `true` | `core` | — |
| `scalar` | 2,272 | `abs`, `add`, `div`, `fma`, `max`, `min`, `mul`, `sqrt`, `sub` | dtype: `bf16`, `f16`, `f32`, `f64` | `core` | mode: `latency`, `throughput` |
| `bit` | 570 | `brev`, `clz`, `lop3`, `popc`, `prmt`, `shf` | dtype: `u32`, `u64` | `core` | mode: `latency`, `throughput` |
| `packed` | 1,138 | `add`, `dp2a_hi`, `dp2a_lo`, `dp4a`, `fma`, `mul` | dtype: `bf16x2`, `f16x2`, `i8`, `i16`, `u8`, `u16`; <br>dtype-b: `i8`, `u8` | `core` | mode: `latency`, `throughput` |
| `sfu` | 768 | `cos`, `exp2`, `log2`, `rcp`, `rsqrt`, `sin` | dtype: `f32` | `core` | mode: `latency`, `throughput`; <br>precision: `approx`, `precise` |
| `convert` | 495 | `cast` | dtype: `bf16`, `f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`; <br>dtype-b: `bf16`, `f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64` | `core` | round: `rd`, `rn`, `ru`, `rz` |
| `tensor` | 6,811 | `mma` | dtype: `b1_and`, `b1_xor`, `bf16`, `e2m1`, `e2m3`, `e3m2`, `e4m3`, `e5m2`, `f16`, `f64`, `i4`, `i8`, `tf32`, `u4`, `u8`; <br>dtype-b: `b1_and`, `b1_xor`, `bf16`, `e2m1`, `e2m3`, `e3m2`, `e4m3`, `e5m2`, `f16`, `f64`, `i4`, `i8`, `tf32`, `u4`, `u8`; <br>accumulator: `f16`, `f32`, `f64`, `i32` | `emulated`, `native` | mode: `latency`, `throughput`; <br>scale: `mx`, `none`, `nv`; <br>scale-exp-a: `-2`, `0`, `2`; <br>scale-exp-b: `-2`, `0`, `1`, `2`; <br>mma-kind: `expanded`; <br>pattern: `alternating_sign`, `ones`; <br>reset-every: `0`, `512` |
| `gemm` | 158,365 | `gemm` | dtype: `bf16`, `c64`, `c128`, `f16`, `f32`, `f64`, `fp4_e2m1`, `fp8_e4m3`, `fp8_e5m2`, `i8`; <br>dtype-b: `bf16`, `c64`, `c128`, `f16`, `f32`, `f64`, `fp4_e2m1`, `fp8_e4m3`, `fp8_e5m2`, `i8`; <br>output: `bf16`, `c64`, `c128`, `f16`, `f32`, `f64`, `i32` | `cublas`, `cublaslt`, `software` | compute: `bf16x9`, `default`, `fast_bf16`, `fast_f16`, `fixedpoint`, `pedantic`, `tf32`; <br>cublas-algo: `default_tensor_op`; <br>scale-mode: `block16`, `block32`; <br>epilogue: `bias`, `bias_gelu`, `bias_relu`, `default`, `gelu`, `relu`; <br>trans-a: `N`, `T`; <br>trans-b: `N`, `T`; <br>batch-mode: `pointer`; <br>emulation-strategy: `eager`, `performant`; <br>mantissa-control: `dynamic`, `fixed`; <br>mantissa-bits: `0`, `64`, `128`; <br>software-policy: `mathematical-reference`; <br>input: `rank1` |
| `blas` | 1,034 | `asum`, `axpy`, `copy`, `dot`, `geam`, `gemv`, `hemm`, `iamax`, `iamin`, `nrm2`, `scal`, `swap`, `symm`, `syrk`, `trmm`, `trsm` | dtype: `c64`, `c128`, `f32`, `f64` | `cublas`, `software` | side: `left`, `right`; <br>uplo: `lower`, `upper`; <br>diag: `nonunit`, `unit` |
| `quant` | 7,220 | `decode`, `encode`, `roundtrip` | format: `bf16`, `e2m1`, `e2m3`, `e3m2`, `e4m3`, `e5m2`, `f16`, `nf4`, `s1`–`s16`, `u1`–`u16` | `core` | scale-format: `e4m3`, `e8m0`, `f32`; <br>distribution: `constant`, `outliers`, `ramp`, `zeros` |
| `quant_gemm` | 11,847 | `gemm` | dtype: `bf16`, `f16`, `f32`; <br>format: `bf16`, `e2m1`, `e2m3`, `e3m2`, `e4m3`, `e5m2`, `f16`, `nf4`, `s1`–`s16`, `u1`–`u16` | `cublas`, `software` | mode: `dequant_gemm`, `end_to_end`, `fused`, `predequant_gemm` |
| `memory` | 573 | `copy`, `cp_async`, `gather`, `pointer_chase`, `read`, `scatter`, `shared_bank`, `shared_copy`, `strided`, `triad`, `write` | dtype: `f32`, `u32` | `core` | — |
| `atomic` | 772 | `add`, `and`, `cas`, `exchange`, `max`, `min`, `or`, `xor` | dtype: `bf16`, `f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64` | `core` | — |
| `collective` | 82 | `ballot`, `barrier`, `block_reduce`, `block_scan`, `reduce_sum`, `scan`, `shuffle`, `sort` | dtype: `f32`, `u32` | `core` | — |
| `workload` | 626 | `attention`, `gelu`, `gelu_tanh`, `layernorm`, `relu`, `rmsnorm`, `rope`, `silu`, `softmax`, `swiglu` | dtype: `bf16`, `f16`, `f32`, `f64` | `core` | causal: `false`, `true` |
| `transfer` | 85 | `d2d`, `d2h`, `h2d`, `managed_first_touch`, `managed_prefetch`, `p2p` | — | `core` | pinned: `false`, `true` |
| `launch` | 6 | `empty`, `graph` | — | `core` | — |
| `sparse` | 82 | `csr_spmm`, `csr_spmv` | dtype: `f32`, `f64` | `cusparse`, `software` | — |
| `fft` | 282 | `c2c`, `c2r`, `r2c` | dtype: `bf16`, `f16`, `f32`, `f64` | `cufft`, `software` | direction: `forward`, `inverse` |
| **Total** | **404,129** | **20 registered families** | — | — | — |

<!-- END GENERATED BENCHMARK CATALOG -->

### Export the literal list of all 404,129 cases

```sh
mkdir -p plans
./bench plan --profile full --software-pairs --output plans/full.jsonl
wc -l plans/full.jsonl
```

The line count is `404129`. Each line contains the canonical family, parameters,
provider, tags and stable case ID. Replaying that `.jsonl` plan does not expand
it again unless `--software-pairs` is explicitly requested. Use this export when
reviewing every correlated shape and control combination; the axes below are
not claims that every displayed value is crossed with every other value.

### Latency, throughput and overhead tests

- Integer, scalar floating-point, bit, packed, SFU and tensor-MMA families have
  distinct `latency` and `throughput` cases. Instruction latency is a one-warp,
  one-block dependency chain with ILP 1; throughput covers ILP 1/2/4/8 and the
  applicable 32/64/128/256/512-thread grids.
- Instruction loops cover 32, 64, 256 and 1,024 iterations. Tensor configurations
  additionally include the fixed 20,000-iteration, 1,056-block throughput case.
- `pointer_chase` measures dependent memory-load latency. Memory bandwidth tests
  separately cover reads, writes, copies, triads, gathers, scatters, striding,
  asynchronous copies, shared-memory copies and bank conflicts.
- Empty-kernel and 1/8/16/128-node graph cases measure launch overhead. H2D, D2H,
  D2D, managed-memory and P2P cases measure transfer behavior rather than ALU
  throughput.
- Every timed family uses the selected sampling policy and records all samples,
  mean, median, standard deviation, CV, MAD, p10/p90, range and drift. An
  amortized reciprocal throughput metric is never relabeled instruction latency.

### Integer, unsigned, floating-point and mixed-operation coverage

- Signed and unsigned integer instructions cover every width from 1 through
  128 bits. Vector-width arithmetic independently covers signed/unsigned widths
  1 through 64. Integer atomics retain every requested width/operation
  combination and report unsupported forms instead of projecting U32 behavior
  onto other types.
- Floating-point scalar and workload paths cover FP16, BF16, FP32 and FP64.
  GEMM/BLAS/FFT also cover real and complex storage where their APIs admit it.
- Packed and mixed integer tests include FP16x2/BF16x2 arithmetic, all signed
  and unsigned DP4A operand pairs, and DP2A low/high combinations.
- Mixed tensor/GEMM tests cover I8/U8, I4/U4, binary AND/XOR, FP16, BF16, TF32,
  FP64, FP8, FP6 and FP4 operands; FP16/FP32/FP64/I32 accumulation; block/MX/NV
  scaling; and native/requested versus explicit emulated/software counterparts.
- Conversions cover the full Cartesian source/destination set of I/U8/16/32/64
  and FP16/BF16/FP32/FP64, with RN/RZ/RD/RU modes where the conversion observes
  rounding. Quantization covers S1–S16, U1–U16, E2M1/E2M3/E3M2/E4M3/E5M2,
  NF4, FP16 and BF16 storage.

### Shape and control axes

| Area | Bundled axes and controls |
|---|---|
| Instruction microbenchmarks | latency/throughput, ILP 1/2/4/8, 32/64/128/256/512 threads, 32/64/256/1,024 iterations |
| Tensor MMA | native/emulated backends, latency/throughput, ILP 1/2/4/8, 32/64/128/256 threads, accumulator and scaling choices, ones/alternating-sign patterns, scale exponents -2/0/1/2 |
| Dense GEMM | cuBLAS/cuBLASLt/software; NN/TN/NT/TT; batch 1/3/4/16 and pointer-batch gaps 7/256; leading-dimension padding 0/3; workspaces 0/64/256 MiB; 4/8/16/32 algorithm requests plus `default_tensor_op`; alpha 0.75/1, beta 0/0.25/0.5, rank-1 validation, block16/block32 scaling and all documented epilogues |
| GEMM dimensions | M: 1, 2, 4, 8, 16, 31, 32, 63, 64, 127, 128, 255, 256, 257, 511, 512, 1,000, 1,024, 2,048, 4,096, 8,192, 16,384; N: 16, 32, 33, 64, 65, 128, 129, 256, 257, 511, 512, 513, 1,001, 1,024, 2,048, 4,096, 8,192, 14,336, 16,384, 28,672; K: 16, 32, 64, 65, 127, 128, 255, 256, 512, 513, 1,003, 1,024, 1,025, 2,048, 4,096, 7,168, 8,192, 16,384 |
| Quantization | groups 1/16/32/64/128/256/1,024; F32/E8M0/E4M3 scales; encode/decode/roundtrip and zero/constant/ramp/outlier distributions; fused/dequant/predequant/end-to-end GEMM modes |
| Memory and systems | 32 through 268,435,456 elements where configured; cache-sized and large sweeps, strides 1/2/3/4/8/16/31/32/33/64/128/256/1,024, shared-memory widths, contention counts and irregular FFT/CSR dimensions |
| Neural reference workloads | softmax, RMSNorm, LayerNorm, RoPE, ReLU, exact GELU, tanh GELU, SiLU, SwiGLU and causal/non-causal attention across FP16/BF16/FP32/FP64, row/column, batch, head and sequence grids |

### Bundled configurations included by `full`

| Configuration | Original cases | With default pairs | Purpose |
|---|---:|---:|---|
| `configs/acceptance.json` | 192 | 235 | Representatives from all 20 families |
| `configs/acceptance_tensor_emulation.json` | 17 | 17 | Exact emulated companions for native acceptance MMA cases |
| `configs/tensor_throughput.json` | 140 | 140 | Fixed-work tensor MMA latency/throughput controls |
| `configs/gemm_throughput.json` | 6 | 12 | FP16/BF16 1,024³, 4,096³ and 8,192³ GEMM plus software pairs |
| `configs/llm_shapes.json` | 372 | 717 | Dense, FP4 and quantized GEMM LLM/decode/prefill shapes |
| `configs/memory_sweep.json` | 408 | 408 | All memory operations over explicit size/stride/work grids |

Optional CUTLASS, Nsight Compute, Compute Sanitizer, NCCL, FlashAttention and
format-specific GGUF/GPTQ/AWQ/EXL integrations are deliberately separate from
the 404,129 core cases. Run `./bench extensions` to see their exact adapter,
external-command or documented-gap status; absence of an external tool is never
reported as a completed core benchmark.

## Build and select the exact GPU

Requirements: Linux x86-64 or WSL2, Python 3.10+, CMake 3.24+, a CUDA-supported
C++17 compiler, CUDA Toolkit 13.5, and a suitable NVIDIA driver. The Python
runner needs no pip packages. cuBLAS/cuBLASLt are required; cuFFT and cuSPARSE
are enabled by default.

```sh
./bench selftest
./bench build --arch 80,89,120a \
  --nvcc /usr/local/cuda-13.5/bin/nvcc --jobs 4

nvidia-smi --query-gpu=index,name,uuid,pci.bus_id --format=csv
```

The workers are `build/bin/archbench_sm80`, `archbench_sm89` and
`archbench_sm120a`. Match the worker to the selected GPU. `--gpu-uuid` resolves
the full UUID among CUDA-visible devices and keeps peer GPUs visible for P2P
cases. `--device N` is also supported; it means a **CUDA-visible ordinal**, which
need not equal an `nvidia-smi` index after visibility/order changes. The two
selectors are mutually exclusive.

### One command for all bundled cases and the complete report

Replace the UUID and choose the matching worker:

```sh
./bench run --profile full --software-pairs --quality stable \
  --gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --binary build/bin/archbench_sm120a \
  --output runs/full_sm120a --timeout 600 --telemetry
```

Software pairs and automatic reports are on by default; they are explicit in
this command for clarity. Do not add `--limit`, family/type filters, or
`--fail-fast` when you want the complete selection. A per-case timeout remains
necessary to finish a campaign with problematic kernels; a timeout is reported,
not counted as a pass. Increase it when a valid large software/reference case
needs more time. These hundreds of thousands of isolated cases can require a
long campaign and considerable report/log storage. No completion-time promise
is inferred from the case count.

Every completed or interrupted run automatically creates:

| Output | Contents |
|---|---|
| `report.html` | Overview, correctness, capability, implementation, stability and metrics |
| `report_details/` | Bounded offline detail pages for large reports; keep with `report.html` |
| `results.csv` | Every selected case, including missing/unavailable cases and all derived metric columns |
| `coverage.json` | Selected, recorded, missing, validated and per-family outcomes |
| `pairs.json` | Requested implementation versus software counterpart coverage |
| `results.jsonl`, `plan.jsonl`, `run.json` | Raw records, immutable requested cases and provenance |
| `logs/`, optional `telemetry.jsonl` | Worker evidence and whole-worker NVML observations |

Small reports are self-contained. Above 5,000 selected cases, the HTML overview
links to offline pages with at most 1,000 cases each. This keeps the full report
usable without embedding hundreds of thousands of detailed records in one page.
Reports do not stop at a display-row limit. Preserve the entire run directory.

Resume with the **same command and `--resume`**. The runner rejects a changed
binary, GPU, environment, timing policy or selected plan. Use a fresh directory
for this release instead of appending to v0.1.0 results.

## Run an identical plan on every GPU

```sh
./bench plan --profile full --software-pairs --output plans/full.jsonl

./bench run --matrix plans/full.jsonl --quality stable --timeout 600 \
  --gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --binary build/bin/archbench_sm80 --output runs/full_sm80 --telemetry

./bench run --matrix plans/full.jsonl --quality stable --timeout 600 \
  --gpu-uuid GPU-yyyyyyyy-yyyy-yyyy-yyyy-yyyyyyyyyyyy \
  --binary build/bin/archbench_sm89 --output runs/full_sm89 --telemetry

./bench run --matrix plans/full.jsonl --quality stable --timeout 600 \
  --gpu-uuid GPU-zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz \
  --binary build/bin/archbench_sm120a --output runs/full_sm120a --telemetry

./bench compare runs/full_sm80 runs/full_sm89 runs/full_sm120a \
  --output reports/architecture_comparison
```

An exact `.jsonl` plan replays unchanged. It is not expanded again unless you
explicitly request `--software-pairs`. Freeze the expanded plan once, as above.
A new source matrix/profile expands pairs before sharding, so related
implementations stay together. Identical case IDs with different SM-dependent
work counts compare useful throughput; an elapsed-time ratio is only used for
equal work/byte scopes. Failed or unavailable results never receive a speedup.

## Start with acceptance or focused throughput

The historical acceptance file retains its **192 original identities**. Default
software expansion selects **235 distinct cases**: some cuBLAS/cuBLASLt
requests share the same software counterpart.

```sh
./bench run --matrix configs/acceptance.json --quality stable \
  --gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --binary build/bin/archbench_sm120a --output runs/acceptance_sm120a

python3 tools/validate_gpu.py --arch 120a \
  --gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --output runs/acceptance_with_sanitizers
```

The acceptance script uses a quick numerical policy and runs Compute Sanitizer
on executed representatives. This is an initial correctness gate; it does not
certify all catalogue cases or establish sustained peak performance.

The two linked `gpu-unlock-ada` experiments have explicit comparable workloads:

```sh
./bench run --matrix configs/tensor_throughput.json --quality stable \
  --device 0 --binary build/bin/archbench_sm89 --output runs/tensor_throughput

./bench run --matrix configs/gemm_throughput.json --quality stable \
  --device 0 --binary build/bin/archbench_sm89 --output runs/gemm_throughput \
  --timeout 600
```

The tensor configuration includes the FP8 E4M3, 1,056 blocks, 256 threads,
eight independent accumulator chains and 20,000-iteration workload, plus 69
additional native/software pairs. The GEMM configuration includes FP16/BF16
inputs with FP32 output at 1024³, 4096³ and 8192³. A structured `rank1` input
pattern allows every output element to be checked against an exact analytic
reference. These match the requested operations and controls; this harness has
its own timing and validation instrumentation. See
[upstream comparison](docs/upstream_comparison.md).

For the linked GEMM test's fixed sampling policy, add:

```text
--quality standard --samples 5 --max-samples 5 --target-cv 0 \
--warmup 10 --warmup-ms 0 --repeats 20
```

That requests ten warmups, five samples and twenty calls per sample. Stable
mode is preferable for a broader campaign when you want variation diagnostics.

## Timing, rates and implementation labels

`--quality stable` uses at least **11** samples, up to **31**, a **50 ms** target
per sample batch, ten warmup calls plus **200 ms** minimum warmup wall time,
and a **3% coefficient-of-variation target**. Calibration respects the repeat
cap. Samples are per-call batch averages, not single-call latency tails.
Stability is measured and reported; it is never guaranteed or created by
removing inconvenient samples. Every sample is retained.

| Quantity | Available presentation / interpretation |
|---|---|
| Time | Primary microseconds, plus milliseconds and nanoseconds |
| Variation | Sample count, mean, standard deviation, CV, MAD, p10/p90, range, first/last-half drift |
| Median interval | Distribution-free order-statistic interval when sample count permits; adaptive stopping and correlation caveats are explicit |
| Floating work | FLOP/s, GFLOP/s, TFLOP/s when the work model is floating arithmetic |
| Other work | Operation-, integer-, bit-pair-, atomic-, conversion-, value- or launch-specific rates |
| Traffic | Bytes/call, logical GB/s and GiB/s when a byte model exists |
| Instruction microbenchmarks | Logical warp MMA counts and cycles per requested instruction when defined |
| Numerical quality | Absolute/relative implementation error; quantization distortion and actual storage cost separately |
| Costs and provenance | Setup, preparation, tuning and validation phases where measured; selected algorithms, timing scope and source/toolchain identity |

Not-applicable or unmeasured fields display **N/A**, not zero. A real exact-zero
error remains zero. Logical bandwidth is not automatically DRAM bandwidth;
small warm buffers can be served from cache. Arithmetic intensity is computed
only when compatible work and byte models both exist.

Implementation labels distinguish requested native PTX/intrinsics, promoted
operations, explicit software kernels, compiler expansions, library emulation
policies, opaque library selection, and requests not executed. At build time,
optional MMA forms are compiled and, when binary utilities are available,
disassembled. In the checked CUDA 13.5.37 build, SM120a INT4/binary requests
expand into INT8 MMA and unpacking; they are labeled compiler emulation. An
accepted probe or direct DMMA opcode is build evidence, not proof of runtime
support or peak throughput on an untested board.

An unavailable native/library request stays visible alongside its software
case. Software execution reproduces a numerical problem on the selected GPU;
it does not simulate another GPU's timing. Internal cuBLAS fixed-point,
BF16x9 and fast-accumulation policies have explicitly labeled mathematical
references, with no claim to reproduce their proprietary internal algorithms.
A missing library heuristic is not proof that the mathematical operation is
impossible. Invalid semantics, resource limits, unimplemented providers and
unsupported device paths are reported separately.

## Coverage inventory

| Selection | Original/requested entries | With default software pairs |
|---|---:|---:|
| `smoke` | 1,360 | 1,608 |
| `standard` | 6,156 | 8,369 |
| `exhaustive` | 315,894 | 399,945 |
| `full` (union of profiles and bundled configurations) | 318,948 | **404,129** |
| `configs/acceptance.json` | 192 | 235 |
| `configs/tensor_throughput.json` | 140 | 140 |
| `configs/gemm_throughput.json` | 6 | 12 |
| `configs/llm_shapes.json` | 372 | 717 |
| `configs/memory_sweep.json` | 408 | 408 |

The 20 families cover signed/unsigned widths 1–128; FP16/BF16/FP32/FP64;
packed arithmetic; conversions; SFUs; tensor MMA including FP8/FP6/FP4;
cuBLAS/cuBLASLt and software GEMM; 16 BLAS operations; 40 portable quantization
encodings; quantized GEMM; memory; atomics; collectives; launch/graph overhead;
transfers; CSR sparse operations; FFT; and neural-network reference workloads.
Complex FP32/FP64 storage is included in GEMM, BLAS and FFT where applicable.

The software GEMM, BLAS, FFT and sparse paths are real GPU implementations,
with the same inputs and semantic controls as their paired requests. Their
reduction order, implementation cost and permitted precision may differ. Native
memory, topology and synchronization features do not gain physical support
through arithmetic emulation; P2P requires a suitable visible peer. External
CUTLASS/NCCL/profiler adapters require their actual tools and are not silently
counted as completed core tests. See [scope](docs/scope.md),
[coverage](docs/coverage.md), [GEMM/BLAS](docs/gemm.md),
[ALU/MMA](docs/alu.md), [software FFT/CSR](docs/systems_software.md),
[quantization](docs/quantization.md), [methodology](docs/methodology.md)
and [record schema](docs/result-format.md).

```sh
./bench list --profile full --count
./bench list --profile full --families gemm --dtypes f64 --count
./bench list --profile standard --no-software-pairs --count
./bench extensions
```

Run performance experiments on an otherwise idle GPU, preserve the recorded
power/clock/environment context, and compare matching plans/policies. Telemetry
covers the entire worker including setup and validation; it is not per-kernel
energy accounting. Use `bench profile`, `bench sanitize` and `bench disassemble`
for separate evidence, retaining instrumentation overhead in its own scope.

## License

ArchBench is original work written from scratch and is licensed under the
[GNU Affero General Public License, version 3 or later](LICENSE). See
[NOTICE](NOTICE) for the copyright and project-license boundary and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for external dependencies and
numeric constants.
