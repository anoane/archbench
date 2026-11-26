# Primary sources and evidence map

**Documentation snapshot reviewed: 25 November 2026.** The project targets
CUDA Toolkit 13.5 and PTX ISA 9.5. Links below point to NVIDIA/vendor/project
primary material. Many are live documentation URLs and can advance after this
snapshot. Record the actual compiler, driver, libraries, binary hash, external
checkout revision, and run metadata when reproducing an experiment.

These sources support API/format/capability decisions. They are not measured
performance evidence for this delivery. The release includes actual CUDA 13.5.37 build evidence and isolated-probe/kernel
disassembly. It has no new GPU execution from the authoring workspace; no source
citation substitutes for runtime correctness, sanitizer or profiling evidence.

## CUDA Toolkit and compiler

| Primary source | Used for | Limit of the evidence |
| --- | --- | --- |
| [CUDA Toolkit 13.5 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/) | Toolkit version, compiler/runtime changes, known issues, release-specific support | Release notes do not prove this project's build or GPU correctness. |
| [Official CUDA Toolkit archive](https://developer.nvidia.com/cuda-toolkit-archive/) | Resolve an actual toolkit release and versioned documentation | A downloaded compiler must still be recorded in run/build metadata. |
| [NVCC compiler driver documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) | Exact `compute_*`/`sm_*` targets, architecture-specific compilation, code generation controls | A target flag is a request; retain compiler logs and generated code. |
| [CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | `cuobjdump` disassembly and resource evidence | Static instructions present anywhere in a binary do not establish a measured case's dynamic path. |

## PTX ISA

Primary specification:
[NVIDIA PTX ISA 9.5](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).

Relevant sections are integer and floating-point arithmetic, packed dot
products, conversions, atomic operations, warp-level matrix instructions,
alternate floating-point formats, block scaling, and target ISA notes.

This specification supports the distinctions used throughout the project:

- PTX is a virtual instruction set; a PTX instruction can lower to multiple
  machine instructions.
- BF16 FMA is available from SM80; dedicated BF16 add/sub/mul instructions
  require SM90 or newer. Older-target CUDA intrinsics can synthesize them.
- The SM80 FP64 MMA path is 8×8×4. The larger 16×8 FP64 shapes have different
  architecture requirements.
- FP8, FP6, FP4, their packing rules, and their supported MMA/scale/accumulator
  combinations are instruction-specific.
- SM120a narrow `mma.sync` instructions are separate from SM100 `tcgen05.mma`.

The specification does not certify a particular board's throughput, enabled
resources, or a specific compiler's final code. The isolated assembler probes
in this repository provide compiler-acceptance evidence and optional static
SASS classification; both are explicitly excluded from measured benchmark
coverage. A direct opcode does not establish runtime support or throughput.

## CUDA mathematical types and conversions

| Primary source | Used for |
| --- | --- |
| [CUDA Math API reference](https://docs.nvidia.com/cuda/cuda-math-api/index.html) | Scalar math, low-precision types, intrinsics, rounding and conversion contracts |
| [Half precision conversion and data movement](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__MISC.html) | Half bit reinterpretation, FP32↔FP16 conversions, directed rounding |
| [BF16 conversion and data movement](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html) | BF16 bit reinterpretation, conversions, packed movement |
| [BF16 arithmetic functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html) | CUDA intrinsic interfaces, separate from inferred hardware implementation |

The portable codec intentionally has its own documented saturation, scale, and
packing policies. A codec test is not automatically a test of every CUDA
conversion mode, NaN payload rule, signed-zero behavior, or subnormal policy.

## cuBLAS and cuBLASLt

Primary API reference:
[NVIDIA cuBLAS 13.5](https://docs.nvidia.com/cuda/cublas/index.html).

| Specific reference | Decisions supported |
| --- | --- |
| [Floating-point emulation](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation) | BF16x9 eligibility, fixed-point FP64 emulation, strategies, retained-mantissa controls and indicators |
| [Narrow precision data types](https://docs.nvidia.com/cuda/cublas/index.html#narrow-precision-data-types-usage) | FP8/FP4 datatype eligibility, scaling modes, hardware scale layouts and restrictions |
| [Complete cuBLAS/cuBLASLt reference](https://docs.nvidia.com/cuda/cublas/index.html) | GEMM variants, mathematical compute policies, transpose/storage conventions, heuristic and epilogue descriptors, BLAS operations |

The CUDA 13.5 emulation documentation identifies BF16x9 support for compute
capabilities 10.0 and 10.3. The suite therefore explicitly rejects that mode
for its requested SM80/SM89/SM120a workers. Fixed-point FP64 emulation has a
different eligibility table, including 8.x and 12.x. The requested policy and
the library's reported retained-mantissa/fallback indicator are separate
evidence fields. Eligibility does not guarantee a particular call used the
emulated implementation.

The raw PTX format matrix must not be substituted for cuBLASLt's API support
matrix. Shape, layout, scaling, output type, accumulator policy, workspace, and
algorithm availability can all change a library result.

## CUTLASS

| Primary source | Used for |
| --- | --- |
| [NVIDIA CUTLASS source repository](https://github.com/NVIDIA/cutlass) | Actual source revision, build configuration and installed kernel inventory |
| [Blackwell functionality and SM120 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html) | Distinction between SM100 and SM120 paths; narrow mixed inputs and block-scaled instruction families |
| [CUTLASS profiler documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html) | Kernel enumeration, operation/problem selection, numerical verification and profiler reports |

CUTLASS is supplied separately by the user, not vendored into this project.
The adapter records its Git revision, dirty state, architecture and kernel
filter. A kernel's existence in documentation is not proof it was built into
the supplied profiler or successfully measured. Structured 2:4 sparse Tensor
Core coverage belongs to actual available CUTLASS/provider cases; core CSR
cuSPARSE tests are a different experiment.

## Numeric formats and NF4

The [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html),
[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html), and
[cuBLAS narrow-precision documentation](https://docs.nvidia.com/cuda/cublas/index.html#narrow-precision-data-types-usage)
are the primary references for the implemented NVIDIA arithmetic encodings and
scale types. They distinguish E2M1 FP4, E2M3/E3M2 FP6, E4M3/E5M2 FP8, FP16,
BF16, TF32, and unsigned exponent-only scale representations.

The NF4 numeric codebook constants match the public bitsandbytes table:

- [bitsandbytes functional definitions](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py)
- [bitsandbytes CUDA kernels](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/kernels.cu)

The relevant MIT notice is reproduced in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). ArchBench's surrounding
encoder/decoder is an independent implementation and uses a different packed
layout. Matching the 16 codebook values does not make this project a decoder
for bitsandbytes artifacts or every NF4 checkpoint.

Named model quantization methods and containers need their own exact format
and implementation sources when added. This document does not invent a single
specification that makes GPTQ, AWQ, GGUF K/IQ, EXL2/EXL3 and generic INT4
interchangeable.

## Runtime, memory and collectives

| Primary source | Used for |
| --- | --- |
| [CUDA runtime memory management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html) | Device/host allocation, transfers, managed-memory and prefetch behavior |
| [CUDA graph management](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html) | Graph construction and replay APIs |
| [CUDA language extensions and atomics](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html) | Thread/warp intrinsics, atomics, architecture and semantic restrictions |
| [CUB device primitives](https://nvidia.github.io/cccl/cub/api/device.html) | Actual device reduction, scan and sorting interfaces from the installed CCCL/CUDA toolkit |
| [NVIDIA cuSPARSE](https://docs.nvidia.com/cuda/cusparse/) | Generic CSR SpMV/SpMM APIs and legal types/layouts |
| [NVIDIA cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) | Transform planning, type/rank/batch support and restrictions |
| [NVIDIA NVML API](https://docs.nvidia.com/deploy/nvml-api/latest/index.html) | Optional power, clock, utilization and device telemetry |

Actual memory traffic, residency, cache hits, link routing and power behavior
require measurements. Logical byte counts in the benchmark are an explicit
traffic model, not evidence of physical DRAM transactions.

## Profiling and correctness instrumentation

| Primary source | Used for | Required distinction |
| --- | --- | --- |
| [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) | Memcheck, racecheck, initcheck and synccheck invocation | A skipped worker is not a successful sanitizer-tested benchmark. Instrumented timing is not normal throughput. |
| [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Counter interpretation, profiling/replay behavior, cache/clock effects, roofline methodology | Counter names and availability vary. Query the installed tool and inspect the actual kernel. |
| [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) | CLI profiling, metric/kernel selection, report export | A trace of initialization/autotuning kernels does not establish the final measured kernel's behavior. |
| [CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) | SASS and resource inspection | Match symbols and runtime kernel evidence before claiming native execution. |

## Optional application providers

| Primary project | Potential external coverage | Core-suite boundary |
| --- | --- | --- |
| [NVIDIA NCCL tests](https://github.com/NVIDIA/nccl-tests) | Multi-GPU collective bandwidth/correctness | No all-reduce result is inferred from the bundled P2P-copy benchmark. |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | Actual optimized attention implementations and their supported forward/backward paths | The included naive/reference attention benchmark is not branded as FlashAttention. |

External applications are not installed or executed merely because a source is
listed here. Their exact revision, build flags, device eligibility, inputs,
verification output and timing scope must accompany any imported result.

## Inspected comparison workloads

The two user-linked gpu-unlock-ada CUDA programs were fetched in full and their
source SHA-256 values, controls and comparison limitations are recorded in
[upstream_comparison.md](upstream_comparison.md). Their code is not redistributed.
The corresponding configurations are tensor_throughput.json and
gemm_throughput.json; these are real requested workloads with software pairs,
not copied performance numbers.
