# Scope and capability boundaries

**Scope snapshot: 25 November 2026. Target toolkit: CUDA 13.5.**

ArchBench is a substantial, finite numerical-compute benchmark project for
SM80, SM89, and SM120a. It combines implemented CUDA kernels, actual CUDA
library calls, portable emulation baselines, and explicit external-provider
adapters. The generated plan is inspectable and reproducible. It is not a claim
to test 100% of CUDA, all possible inputs, every machine instruction, every
matrix shape, every quantization method, or every proprietary library kernel.

This distinction matters to coverage accounting. A planned case, an accepted
compiler probe, an unsupported operation, a measured result, and a numerically
validated result are different things. A skipped or rejected operation never
becomes a successful benchmark with zero throughput. The plan defines a
finite denominator; unfinished or unsupported cases remain visible.

The CUDA source targets CUDA 13.5 headers and libraries. All three complete
workers were compiled and linked with NVCC 13.5.37 during this release. Static
probe/kernel disassembly and host checks are separate evidence. The authoring
workspace has no NVIDIA GPU; run acceptance and sanitizers on every actual
target before relying on new performance results. See validation.md for the
exact checked build and the historical user-supplied GPU evidence.

## What is implemented

“Implemented” below means executable source is included. It does not mean a
particular architecture/library accepts every generated parameter combination,
or that a real GPU result was obtained in the authoring environment.

| Area | Included core implementation | Boundary |
| --- | --- | --- |
| Integer arithmetic | Signed and unsigned widths **1 through 128**, arithmetic, multiply-add, divide/remainder, comparisons, shifts, bit logic | Standard 8/16/32/64-bit routes request PTX; other widths use two-limb software arithmetic. There is no native-width claim for arbitrary integers. |
| Integer streams | Every width 1..64, signed/unsigned, modulo arithmetic and bit operations over arrays | Stored in 64-bit containers; a separate memory-inclusive baseline from register recurrences. |
| Floating scalar arithmetic | FP16, BF16, FP32, FP64 arithmetic, FMA, division, square root, min/max/absolute value | Selected operations and rounding paths; no arbitrary-precision floating implementation. |
| Packed arithmetic | FP16×2, BF16×2, DP4A, DP2A low/high, mixed signedness | Packed integer dot products are different from full matrix multiplication. |
| Bit manipulation and special functions | Population count, leading zeros, bit reversal, permutation, ternary logic, funnel shift; selected approximate/precise SFUs | Not every PTX/SASS instruction or modifier. SFU recurrences include documented domain-maintenance arithmetic. |
| Conversion | All ordered pairs of 12 standard scalar types; applicable directed rounding controls | Typed read/convert/write streams. Input values are finite and range-safe; this is not exhaustive exception or large-integer rounding-boundary validation. |
| Dense warp MMA | FP16/BF16/TF32/FP64, signed/unsigned INT8/INT4, binary operations, FP8, all 25 FP4/FP6/FP8 narrow pairings, MX and NVFP4 variants | Register-resident instruction baselines. Uniform fragments validate arithmetic and scaling, not every fragment layout or a full matrix-loading pipeline. |
| cuBLAS GEMM | Actual GEMM, strided-batched GEMM, pointer-array batched GEMM; real, complex, integer, and precision-policy requests | The installed library decides legal combinations and implementation. Requests do not prove Tensor Core selection. |
| cuBLASLt GEMM | Actual descriptors, heuristic enumeration and timing, selected epilogues, FP8 and supported block-scaled FP8/FP4 requests | Heuristic candidates are not every internal algorithm. FP6/MXFP4 are not silently presented as implemented cuBLASLt API paths. |
| BLAS beyond GEMM | Selected vector operations, GEMV, GEAM, triangular solve/multiply, SYRK, SYMM, HEMM | The implemented list is explicit in [gemm.md](gemm.md); it is not all BLAS/cuBLAS routines. |
| Quantization codecs | Signed/unsigned integer encodings at widths 1..16; FP4 E2M1; FP6 E2M3/E3M2; FP8 E4M3/E5M2; NF4; FP16/BF16 | **40 numeric encodings**, with a documented portable packing, group scales, offsets, rounding, and saturation policy. They are not 40 model-checkpoint formats. |
| Quantized GEMM baselines | Fused software dequantization + SIMT GEMM; dequantization + cuBLAS; pre-dequantized GEMM; encode + decode + GEMM | Activations may be rounded to FP16/BF16 but remain stored as FP32 in this baseline. These paths do not claim native W4A16/W8A16 Tensor Core execution. |
| Memory and synchronization | Streaming, strided, permutation gather/scatter, dependent pointer chase, shared-memory bank experiments, `cp.async`, shuffle/ballot/barrier | Working-set changes are measured; cache-level attribution requires counters. Direct TMA/distributed-shared-memory tests are not included. |
| Atomics | Integer and floating add plus selected CAS, exchange, min/max, and bit operations; contention sweeps | Subword integer atomics and some floating operations use explicit emulation. API availability does not establish one-instruction execution. |
| Collectives | Block collectives and CUB reductions/scans/sort | These are intra-GPU operations. They are not multi-GPU NCCL collectives. |
| Composite kernels | Softmax, RMSNorm, LayerNorm, RoPE, selected activations/SwiGLU, attention reference | Portable reference/baseline kernels; attention is not FlashAttention and has no implemented backward pass. |
| Launch and CUDA Graphs | Empty launch and graph replay overhead | Reported as measurement-only when no numerical correctness assertion is meaningful. |
| Transfers | Pageable/pinned H2D/D2H, D2D, managed prefetch/first touch, P2P when available | Requires the actual topology and driver capability. No NVLink/PCIe generation or bandwidth is inferred solely from SM version. |
| Sparse arithmetic | Actual cuSPARSE CSR SpMV/SpMM plus matching CUDA software kernels | General CSR sparsity is not structured 2:4 Tensor Core MMA. |
| FFT | Actual cuFFT plus CUDA radix-2/DFT C2C/R2C/C2R counterparts, multiple types, ranks and batches | Exact dtype/shape/device restrictions remain library outcomes. Not all cuFFT planning modes or callbacks are covered. |
| Evidence and reporting | Per-case outcomes, timing samples, validation metadata, resume/provenance checks, comparison reports, telemetry, instrumentation adapters | Instrumented timing, compile-only results, external summaries, and normal timing samples remain separately classified. |

Detailed operation and parameter inventories are in [alu.md](alu.md),
[gemm.md](gemm.md), and [systems.md](systems.md). The catalogue defines the
default profiles; an explicit matrix can select other implemented combinations.

## Architecture facts and this project's native paths

A compute capability identifies an architecture programming target. It does
not uniquely determine active SM count, clocks, memory bandwidth, power limits,
PCIe connectivity, or board-level features. In particular, an SM80 board must
not be assigned A100 throughput or interconnect properties solely from `sm_80`.
Compare the reported properties of the actual cards.

The following table distinguishes raw arithmetic/instruction paths from
library eligibility. “Native request” means the suite requests the relevant
PTX or CUDA intrinsic. Establishing the final machine path still requires
kernel-specific SASS or profiler evidence.

| Feature | SM80 | SM89 | SM120a | Interpretation |
| --- | --- | --- | --- | --- |
| FP32 scalar arithmetic | Native CUDA arithmetic | Native CUDA arithmetic | Native CUDA arithmetic | Ordinary FP32 arithmetic is not generically “emulated.” |
| FP64 scalar arithmetic | Native CUDA arithmetic | Native, with product-dependent low throughput | Native, with product-dependent low throughput | Low throughput is not software emulation. Do not apply one fixed throughput ratio to every product. |
| FP16/BF16/TF32 dense MMA | Included | Included | Included | Accumulator/type restrictions are explicit. TF32 input/compute policy is not interchangeable with strict FP32 arithmetic. |
| FP64 8×8×4 MMA request | Compile-probed; DMMA observed | Compile-probed; DMMA observed | Compile-probed; DMMA observed | CUDA 13.5.37 static build observation; runtime validation remains required. Explicit scalar software counterparts are available on each target. |
| FP8 E4M3/E5M2 dense MMA, FP32 accumulator | Explicit emulation baseline | Native request | Native request | Four raw FP8 A/B pairings do not imply every cuBLASLt pairing is accepted. |
| FP4/FP6 mixed dense MMA | Explicit emulation baseline | Explicit emulation baseline | Architecture-specific native request | All 25 narrow input pairings are implemented at the warp-MMA level. |
| MX block-scaled narrow MMA / NVFP4 | Explicit emulation baseline | Explicit emulation baseline | Architecture-specific native request | Register packing and scale representation depend on the exact instruction family. |
| INT8 signed/unsigned dense MMA | Included | Included | Included | Four A/B signedness combinations, with INT32 accumulation. |
| INT4 and binary MMA | Direct IMMA/BMMA observed | Direct IMMA/BMMA observed | Accepted compiler expansion into INT8 IMMA | Exact optional forms are compile-probed and disassembled where tools are available. Compiler expansion is labeled emulation; scalar counterparts remain separate. |
| BF16 FMA | Available from SM80 | Available | Available | BF16 add/sub/mul are a different PTX feature: their dedicated instructions require SM90; headers synthesize older-target paths. |
| cuBLAS BF16x9 FP32 emulation | Unsupported by explicit suite gate | Unsupported by explicit suite gate | Unsupported by explicit suite gate | CUDA 13.5 documents this mode for compute capabilities 10.0 and 10.3, outside the requested targets. |
| cuBLAS fixed-point FP64 emulation | Library request | Library request | Library request | Documented architecture eligibility does not guarantee emulation was used for a particular call. Retained-mantissa telemetry records fallback evidence. |

These distinctions follow the [PTX ISA](sources.md#ptx-isa),
[cuBLAS precision/emulation documentation](sources.md#cublas-and-cublaslt),
and [CUTLASS Blackwell functionality](sources.md#cutlass). Exact source/runtime
conditions are described in the individual module documents.

### SM120a is not SM100 with a different name

This project's Blackwell workstation/consumer path uses architecture-specific
warp-synchronous `mma.sync` narrow and block-scaled instructions. The SM100
datacenter `tcgen05.mma` and tensor-memory architecture is a separate path.
The suite does not turn an SM100 example into an SM120a capability claim.
The `a` compilation target matters; a generic `sm_120` binary alone does not
establish that architecture-specific code was generated.
[Source: CUTLASS and PTX](sources.md#cutlass).

### Precision labels do not identify the executed library kernel

The following are separate experimental choices:

1. The stored input/output format, such as FP32, FP16, E4M3, or E2M1.
2. The arithmetic/accumulator policy, such as strict FP32, TF32, FP16
   accumulation, FP32 accumulation, or library emulation.
3. The selected library algorithm, tile/epilogue, workspace and layout.
4. The actual executed instructions and preprocessing costs.

cuBLAS and cuBLASLt may reject a combination, choose a fallback, or choose
different kernels for different shapes. A 5×5 raw narrow-MMA input matrix does
not establish a 5×5 cuBLASLt datatype API. A successful FP64 library call does
not by itself establish fixed-point emulation. The result's requested policy,
reported emulation indicator, and measured kernel evidence must be read
together. See [gemm.md](gemm.md).

## Numeric formats and model quantization are separate layers

FP4 in the implemented narrow path is **E2M1**, whose sign, exponent, and
mantissa occupy four bits in total. **E4M3 is eight-bit**, not FP4. The two
implemented FP6 formats are E2M3 and E3M2. E8M0 is used as a scale format;
it is not treated as an ordinary signed scalar floating arithmetic type.
NF4 is a 16-entry nonlinear codebook, not an exponent/mantissa FP4 format.
[Format references](sources.md#numeric-formats-and-nf4).

The software codec has an explicitly documented packing: low bit first within
group-aligned storage, plus separate scales and offsets. Signed integer
encodings use two's complement; signed one-bit values are `{-1,0}`, not a
bipolar `{-1,+1}` model quantizer. Tensor binary AND/XOR-plus-popcount is yet
another operation. Finite saturation and ties-to-even are deliberate codec
policies, not a promise to reproduce every CUDA conversion mode.

Names such as GPTQ and AWQ identify quantization approaches and associated
implementations. GGUF is a container with multiple concrete quantized layouts;
K/IQ families include additional block structure and/or codebooks. EXL2/EXL3
also require their actual packing and runtime. Generic INT4 arithmetic cannot
be relabeled as any of those formats.

Similarly, the NF4 codebook constants match the public bitsandbytes table, but
this project's packing is not bitsandbytes artifact compatibility. The source
and MIT notice are retained in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

To benchmark a named model format, provide its actual producer/version, packed
layout, scale/zero-point/codebook metadata, reference decoder, and compatible
kernel/runtime. The external-command adapter preserves that provider's
invocation and raw evidence; it does not infer numerical correctness from a
successful process exit.

## External providers and explicit gaps

| Requested area | Current route | What is not claimed |
| --- | --- | --- |
| Optimized dense, mixed-input, grouped, and block-scaled GEMM beyond core paths | Version-recorded CUTLASS build/profiler adapter | Availability of every CUTLASS kernel on every target, or optimal tuning. The actual installed kernel catalogue is authoritative. |
| Structured 2:4 sparse Tensor Core MMA/GEMM | CUTLASS profiler, when the selected revision and target supply the kernels | The core CSR tests do not count as structured sparse Tensor Core coverage. |
| FlashAttention forward/backward | Actual external implementation via external-command adapter | The included attention baseline is not FlashAttention, FlashAttention-2/3, or a backward implementation. |
| MoE grouped expert GEMMs/routing and larger fused model pipelines | External implementation/CUTLASS kernels where available | An end-to-end MoE engine, token router, or complete quantize→GEMM→SwiGLU→requantize pipeline is not bundled. |
| NCCL multi-GPU collectives | External NCCL tests | A P2P copy is not all-reduce, all-gather, or reduce-scatter. |
| GPTQ/AWQ/GGUF K/IQ/EXL2/EXL3 and other named formats | Explicit external runtime and format adapter | Automatic compatibility with model files or every existing quantization method. |
| Nsight Compute / SASS evidence | Instrumentation and disassembly adapters | One fixed set of metric names on every architecture, or static opcode presence proving a particular timed path. |
| Compute Sanitizer | Instrumentation adapter with numerical outcome checks | Sanitizer success for an unsupported/skipped worker case, or sanitizer timing used as normal throughput. |
| TMA tensor-map operations, distributed shared memory, clusters, all asynchronous pipelines | Documented gap in direct core benchmarks; selected external kernels may exercise them | Dedicated isolated coverage or counters without actual evidence. |
| FP128, arbitrary-precision floating arithmetic, all decimal/posit/custom formats | Not implemented | FP64 plus a label is not FP128 emulation. |
| All PTX/SASS opcodes, all cache policies, all memory-model litmus tests, all rounding and exception combinations | Not implemented exhaustively | Formal ISA conformance, complete race freedom, or exhaustive mathematical correctness. |
| All cuBLAS/cuBLASLt/cuFFT/cuSPARSE APIs and hidden algorithms | Not implemented exhaustively | API-complete vendor-library regression testing. |
| Graphics, ray tracing, video encoding/decoding, optical flow | Outside this numeric CUDA scope | A complete GPU graphics/media benchmark. |

External-provider enumeration is inventory, not measured coverage. Imported
CUTLASS rows require the provider's success, passed verification, and a positive
runtime before being classified as validated measurements. Their timing scope
remains CUTLASS's reported statistic rather than an invented ArchBench sample
distribution. Raw external-command results remain external evidence.

## What a defensible comparison can conclude

For a selected finite plan, the suite can establish which requested cases the
installed software/hardware accepted, which numerical checks passed, the
distribution of measured execution times, and the recorded execution evidence.
It can compare those same cases across the actual three GPUs while showing
unsupported cases and gaps.

It cannot derive the performance of unimplemented formats, all LLM inference
engines, or unseen model shapes from a uniform MMA loop. It cannot turn
theoretical hardware capability into a measured result, or turn host-only
validation into a GPU pass. Use the register microbenchmarks, full GEMMs,
software quantization baselines, memory tests, and external application
providers together, with their timing boundaries intact.

## Version 0.2 software pairing

The runner selects explicit tensor, GEMM, BLAS, FFT, CSR and decoded quantized
GEMM software counterparts by default. It preserves the original native/library
request, its rejection or failure, and the numerical reference as separate
records. Exact JSONL replay remains unchanged unless expansion is requested.
`full` combines every bundled profile/configuration; it currently selects
404,129 distinct cases with software expansion. These are requests, not passed
tests. Software reference kernels execute on the selected GPU and do not
simulate another architecture's timing. Internal library emulation policies
receive explicitly nonequivalent mathematical references. Native physical
features such as peer connectivity cannot be created by arithmetic emulation.
See release_0_2_0.md and systems_software.md for the added implementations.
