# GEMM and cuBLAS experiments

`src/gemm.cu` implements the `gemm` and `blas` families with cuBLAS/cuBLASLt and
real CUDA software baselines. Every backend creates actual device matrices,
checks output against CPU arithmetic, and begins performance measurement only
after correctness passes. GEMM also checks the final measured output. A library
request with no available implementation remains explicitly unsupported. A
separate software case can evaluate its matching numerical operation; it does
not turn the rejected library request into a successful native result.

All three complete workers in this revision compiled and linked with CUDA
13.5.37. The user-supplied `reports/A.zip` contains four real 192-case acceptance
runs on SM80 and SM120a with the earlier compilation fixes. It provides GPU
evidence for that earlier source, including the failures discussed in the
review. The new software and tuning changes have host checks and real CUDA
build evidence but still require GPU execution. See validation.md for the
recorded gates. No SM89 GPU run was supplied.

## Implemented GEMM interfaces

| Backend | Real implementation |
| --- | --- |
| `cublas` | `cublasGemmEx`, `cublasGemmStridedBatchedEx`, and `cublasGemmBatchedEx` |
| `cublaslt` | `cublasLtMatmul` with separate immutable C and output D; each candidate is correctness checked before warmup and repeated timing; fastest validated median is selected |
| `software` / `emulated` | Custom CUDA SIMT GEMM with 16×16 shared tiles, packed input decoding, the same generated tensors and tiled scale arrays, separate immutable C and output D |

All backends use column-major matrices. Leading dimensions are
computed from the actual stored shape after transpose selection. `--ld-pad`
adds padding to each leading dimension; `--batch-gap` adds elements between
successive matrix allocations. The integer strides in result metadata are in
elements, not bytes.

For a result `D = alpha * op(A) * op(B) + beta * C`, supported command parameters
are:

| Parameter | Values / meaning |
| --- | --- |
| `--m`, `--n`, `--k` | Positive mathematical dimensions; default 1024 |
| `--batch` | Positive batch count; default 1 |
| `--batch-mode` | `strided` or `pointer`; actual pointer-array API dispatch is implemented by classic cuBLAS. Software evaluates the same regular batch addresses and explicitly excludes pointer-array API overhead |
| `--dtype`, `--dtype-b` | A and B input types; B defaults to A |
| `--output-dtype` | C and D storage type |
| `--trans-a`, `--trans-b` | `N`, `T`, or `C` (conjugate transpose) |
| `--alpha`, `--beta` | Real scalar components; default 1 and 0 |
| `--alpha-imag`, `--beta-imag` | Imaginary components for complex operations only |
| `--ld-pad`, `--batch-gap` | Nonnegative padding, measured in elements |
| `--compute` | `default`, `pedantic`, `f16`, `f16_pedantic`, `tf32`, `fast_f16`, `fast_bf16`, `bf16x9`, `fixedpoint` |
| `--workspace-mib` | User-supplied workspace; default 32 MiB |
| `--cublas-algo` | Classic cuBLAS only: `default` (default) or `default_tensor_op`; passed to single, strided, and pointer-batch GemmEx calls and recorded explicitly |
| `--validation-samples` | Number of output elements checked against CPU dot products; default 128; all output elements are checked if the matrix has no more than this many elements |
| `--allow-atomics` | Classic cuBLAS atomics mode; default false |
| `--no-reduced-reduction` | Request accumulator-precision reductions in classic cuBLAS |
| `--sm-count` | Optional classic-cuBLAS SM-count target; zero keeps the library default |
| `--software-policy` | `same-request` (default) or `mathematical-reference`; described below |
| `--input-pattern` | `seeded` (default, varied stored values) or bounded `rank1` throughput pattern with an analytic check of every output |

Not every combination is a legal library operation. Examples include mismatched
real/complex types, integer output with an incompatible compute type, or narrow
precision with an unsupported transpose. These combinations are resolved by the
library, not by a guessed hardware-support table.

### Storage types

| Name | Storage | Applicable backend |
| --- | --- | --- |
| `f16` | IEEE half | cuBLAS, cuBLASLt, software |
| `bf16` | Bfloat16 | cuBLAS, cuBLASLt, software |
| `f32` | IEEE binary32 | cuBLAS, cuBLASLt, software |
| `f64` | IEEE binary64 | cuBLAS, cuBLASLt, software |
| `c64` | Two binary32 components | All three, where the exact operation is accepted |
| `c128` | Two binary64 components | All three, where the exact operation is accepted |
| `i8` | Signed eight-bit input | All three, with `i32` output in software |
| `i32` | Signed 32-bit storage | Software input/output; library requests only where accepted, with no general INT32-input Tensor Core claim |
| `fp8_e4m3` | E4M3 FP8 input | cuBLASLt and software |
| `fp8_e5m2` | E5M2 FP8 input | cuBLASLt and software |
| `fp4_e2m1` / `nvfp4` | Packed E2M1 FP4 input, normally block16 scales | cuBLASLt and software |

The FP4 alias does not imply an interchangeable file format with model
checkpoints. Input codes and hardware-layout scale arrays are generated for
this experiment. FP6 and MXFP4 are not claimed as cuBLASLt API implementations.
Use the explicitly labelled custom or CUTLASS cases elsewhere in the suite for
those formats.

### Compute policies and emulation

`default` requests the ordinary library policy for the storage type. `pedantic`
requests the corresponding pedantic compute type. The fast FP32 modes request
the documented automatic down-conversion policies. An enabled policy is not
proof that a Tensor Core kernel was selected.

Classic `--cublas-algo default` passes `CUBLAS_GEMM_DEFAULT`.
`--cublas-algo default_tensor_op` passes `CUBLAS_GEMM_DEFAULT_TENSOR_OP`.
These are actual API controls with their requested enum recorded; neither proves
the library selected a particular internal kernel or hardware instruction.
cuBLASLt rejects the nondefault classic control because it selects algorithms
through its own descriptor-local heuristic API. A software pair retains this
parameter as source-request provenance and records that its custom kernel does
not apply the library algorithm control.

For current CUDA 13.5 library support, NVIDIA documents BF16x9 on compute
capabilities 10.0 and 10.3, whereas fixed-point FP64 emulation includes 8.x and
12.x. Both library backends explicitly reject `--compute bf16x9` on devices outside
that documented set before dispatching a library operation. SM80, SM89, and
SM120a therefore return `status=unsupported`, `validation=not_run`, and
`execution=not_executed`, with no timing samples for this compute policy.
The result records `compute_policy_support=api_available_architecture_unsupported`,
the CUDA 13.5 capability baseline, the supported compute capabilities, and the
source URL. The policy API's availability does not establish supported BF16x9
execution on these targets, and a successful library fallback would not count
as BF16x9 coverage. See the official
[floating-point emulation documentation](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation).

For `bf16x9` and `fixedpoint`, `--emulation-strategy` accepts `eager`,
`performant`, or `default`. Fixed-point mode additionally supports:

| Parameter | Meaning |
| --- | --- |
| `--mantissa-control dynamic` | Library chooses precision at runtime; default |
| `--mantissa-control fixed` | Request a fixed retained mantissa width |
| `--mantissa-bits` | Maximum/fixed mantissa count; zero keeps the library default |
| `--mantissa-offset` | Offset used by dynamic precision control |

For `fixedpoint`, both library backends request the library's retained-mantissa
indicator. Results include `fixedpoint_indicator_validation_raw`,
`fixedpoint_indicator_timing_raw`, `fixedpoint_indicator_raw`, and
`fixedpoint_used_last_call`. The old `fixedpoint_reported_mantissa_bits` field
is preserved as a compatibility alias for the raw integer and is explicitly
labelled as an un-decoded value. Classic cuBLAS additionally records the maximum
mantissa setting returned by its getter.
The suite preserves the raw integer indicator and derives the last-call flag
from its sign (`-1` means false, positive means true, other values mean unknown).
The supplied CUDA 13.5 runs returned the positive value `4247`. Its precise
interpretation has not been established in this review; do not describe it as
4,247 bits of numerical precision or as instruction-level proof. This evidence
is separate from the requested compute policy and does not describe every
kernel inside a batched call. Phase labels distinguish the premeasurement check
from the last measured call. Neither the CUDA 13.5 header declarations nor the
reviewed public documentation explains the observed `4247` encoding; the suite
does not invent a decoder or infer a literal precision from it.

The library can allocate additional internal fixed-point workspace beyond the
provided workspace. The suite records the supplied workspace but does not claim
it is an absolute cap on all cuBLAS allocations. Reduced precision choices that
fail the FP64 reference are recorded as validation failures and are not promoted
to valid throughput results.

## cuBLASLt-specific controls

`--algorithms` asks for up to this many heuristic candidates (default 16, capped
at 128). Each available candidate first runs against the shared reference with
the unchanged tolerance. Only a passing candidate proceeds to warmup,
calibration, and repeated timing. Selection uses the median of its normalized
samples. The selected candidate is checked again before and after independent
performance measurement. No tuning sample is reused as a final performance
sample.

| Control | Default | Meaning |
| --- | --- | --- |
| `--heuristic-index` | `-1` | Search all returned candidates; a nonnegative value selects only that descriptor-local index, which must still pass correctness |
| `--tune-warmup` | `3` | Untimed warmup calls for each passing candidate |
| `--tune-samples` | `5` | Independent timed candidate batches |
| `--tune-repeats` | `3` | Minimum calls in each tuning batch and initial calibration count |
| `--tune-target-ms` | `1` | Target elapsed time per tuning batch |
| `--tune-max-repeats` | `256` | Maximum calls per tuning batch |

The heuristic index is local to a particular descriptor, library version,
workspace, and returned heuristic list. It is not a stable global kernel ID.
An unavailable explicit index produces an unsupported result. If every
executable candidate fails its numerical check, the case is
`validation_failed`, with no selected candidate and no valid throughput result.
An unsupported descriptor remains unsupported.

Results record the number of candidates, selection times, chosen heuristic
index, actual workspace requirement, and available algorithm ID, tile,
split-K, reduction, and stage configuration. This is a search over returned
heuristics, not a claim to have enumerated every internal cuBLASLt algorithm.
`candidate.<index>.*` retains each candidate's status, configuration, workspace,
validation errors, bounded first/worst mismatch evidence, diagnostic validation
call time, and any subsequent timing samples. Rejected numerical candidates are
excluded from performance timing and selection. Their validation-call time is
explicitly a diagnostic, not accepted performance. `candidate_ms` is retained
as a compact list of valid tuning medians.

This addresses the supplied BF16 `bias_gelu` case where one selected candidate
failed the shared reference and another candidate passed. The suite selects
from candidates that pass; it does not loosen the bound or assume the failed
candidate's internal rounding behavior has been proved.

`--allow-atomics`, `--no-reduced-reduction`, and `--sm-count` are implemented as
classic-cuBLAS handle controls. A nondefault request for them on this Lt backend
is rejected explicitly, rather than silently ignored.

`--epilogue` accepts `default`, `bias`, `relu`, `gelu`, `bias_relu`, or
`bias_gelu`. Bias values vary across rows and are included in the reference.
The bias storage type follows the selected output type, except that narrow
inputs with FP32 output use the documented BF16 bias type. Library-rejected
epilogues remain explicit unsupported cases.

### FP8 and block-scaled input

| Requested path | Options |
| --- | --- |
| Tensor-scaled FP8 | `--dtype fp8_e4m3` or `fp8_e5m2`, optional distinct `--dtype-b`, `--scale-mode tensor` |
| MXFP8-style block scales | FP8 A/B, `--scale-mode block32 --trans-a T --trans-b N --batch 1` |
| NVFP4 | `--dtype fp4_e2m1 --scale-mode block16 --trans-a T --trans-b N --batch 1` |

Tensor scaling uses real device-resident FP32 scale values supplied by
`--scale-a` and `--scale-b`. Block scaling stores a varying 0.5, 1, or 2 scale
for each block in the actual tiled scale layout. FP4 uses E4M3 scales; block32
FP8 uses unsigned E8M0 scales. The CPU reference applies the same block at each
reduction coordinate. Scales are not uniformly set to one, which would hide
scale-layout mistakes.

`--fast-accum true` requests the FP8 fast accumulation descriptor setting.
Accuracy is still checked. `--output-dtype` may select `f16`, `bf16`, or `f32`;
FP8/FP4 output quantization is not implemented in this module.

The scale-layout formula and legal datatype/scale combinations are documented
by NVIDIA in
[narrow precision data types](https://docs.nvidia.com/cuda/cublas/index.html#narrow-precision-data-types-usage).

## Software GEMM and matching numerical references

`--backend software` (alias `emulated`) launches an actual GPU kernel. Each
thread computes one output; 16×16 thread blocks reuse reduction tiles in shared
memory. Inputs are read from their real device storage. FP4 uses the same low
nibble first packing as the library input generator; mixed E4M3/E5M2 operands
are decoded independently. Tensor FP32 and tiled UE4M3/UE8M0 scale arrays are
loaded from device memory using the same logical scale coordinates as the CPU
reference. Output conversion happens after alpha/beta, bias, and activation.

Software supports all declared input storage types and `f16`, `bf16`, `f32`,
`f64`, `c64`, `c128`, and `i32` outputs in the combinations below. FP8/FP4 output
quantization and `i8` output are explicitly unsupported. Integer and floating
operands cannot be mixed. Complex operations require consistent c64 or c128
storage; ReLU/GELU are not applied to complex values. Integer GEMM requires two
integer operands and i32 output; a bound proves its seeded integer result fits
INT32. Integer GELU is unsupported.

| Requested compute policy | Software `same-request` implementation |
| --- | --- |
| `default`, `pedantic` | FP32 SIMT FMA for ordinary real/c64 inputs; FP64 SIMT FMA for f64/c128; exact bounded integer arithmetic for integer inputs |
| `f16`, `f16_pedantic` | Actual scalar half FMA and half accumulator; f16 A, B and output required |
| `tf32` | Explicit round-to-nearest-even TF32 operand conversion followed by FP32 SIMT FMA; f32 A/B/output required |
| `fast_f16` | Explicit IEEE half operand rounding followed by FP32 SIMT FMA; f32 A/B/output required |
| `fast_bf16` | Explicit BF16 operand rounding followed by FP32 SIMT FMA; f32 A/B/output required |
| `bf16x9`, `fixedpoint`, or `fast-accum=true` | Requires `software-policy=mathematical-reference`; the library-specific algorithm is not emulated |

`same-request` matches the declared storage and arithmetic precision. It does
not promise bitwise agreement with a library reduction order, internal
epilogue implementation, or selected instruction sequence. Results state this
scope explicitly. FP32 down-conversion modes are reproducible arithmetic
models, not claims of native Tensor Core execution.

`--software-policy mathematical-reference` supplies a corresponding numerical
baseline even when the requested library-specific policy cannot execute. It
preserves stored inputs, scales, shapes, leading dimensions, batch strides,
transpose/conjugation, alpha/beta scalar storage, bias, activation, and output
storage. It computes the dot product in ordinary FP32, FP64, or bounded integer
arithmetic as appropriate. It records the requested compute policy, actual
implementation policy, and `policy_equivalence=false`. A successful FP64 SIMT
reference for a fixed-point request does not establish fixed-point execution;
a successful FP32 reference for BF16x9 does not establish BF16x9 capability.

Software block scaling can also evaluate transpose and batch shapes that the
Lt path rejects. For those extensions, the same row/column scale arrays are
reused in every batch, and this is recorded. FP4 block16 and FP8 block32 scale
encodings remain distinct. Library workspace, atomics allowance, SM target,
heuristic and tuning controls have no software-kernel scheduling equivalent;
their inapplicability is recorded. The mathematical operation remains paired,
but these are not equal-overhead API implementations.

This kernel is a clear, tiled SIMT baseline. Its throughput measures actual
software decoding and SIMT execution on the selected GPU. It is not a prediction
of an unavailable native FP4/FP8 Tensor Core implementation or an optimized
CUTLASS kernel.

## Large GEMM throughput pattern

`--input-pattern rank1` uses independent implementations of the deterministic
workload from the user-supplied
[GEMM throughput example](https://github.com/JustAResearcher/gpu-unlock-ada/blob/main/docs/evidence/rtx4090-pr1-stock/gemm_bench.cu):
`A[row,col]=(row%13-6)/8`, `B[row,col]=(col%11-5)/8`, so
`D[row,col]=K*((row%13-6)/8)*((col%11-5)/8)`.

It requires matching f16 or bf16 A/B, f32 output, NN, batch 1, alpha 1, beta 0,
default compute and epilogue, and no input scaling. K is bounded at 65,536 so
every partial sum of these binary fractions fits exactly in FP32. The validator
checks every logical output against the exact analytic value, using bounded
host chunks and zero tolerance. Non-square shapes and leading-dimension padding
are supported. Padding is excluded from the logical output check. Checks run
before and after timing and are not included in operation samples.

`configs/gemm_throughput.json` supplies 1024, 4096 and 8192 square shapes for
both input types, with `cublas-algo=default_tensor_op` to match the linked
program's explicit GemmEx algorithm argument. Software pairs evaluate the same
data and output. This highly
structured workload is useful for sustained throughput and full-output checking;
the default `seeded` pattern remains necessary for varied numerical cases.

## BLAS beyond GEMM

Use `--family blas --dtype f32|f64|c64|c128` with `--backend cublas` or
`--backend software` and one of these operations:

| Level | Operations |
| --- | --- |
| 1 | `axpy`, `scal`, `dot`, `nrm2`, `asum`, `iamax`, `iamin`, `copy`, `swap` |
| 2 | `gemv` |
| 3 / extensions | `geam`, `trsm`, `trmm`, `syrk`, `symm`, `hemm` |

The software BLAS path shares the exact setup, stored input values, increments,
matrix controls, CPU references and reset/timing code with the library path.
Vector reductions use a 256-thread block with a fixed reduction tree; matrix
outputs use SIMT loops; triangular solve uses sequential substitution per
independent right-hand side. These are measured GPU baselines, not optimized
library substitutes. A cuBLAS handle may still be created for common setup and
version metadata, but the software operation's timed callback contains no
cuBLAS operation.

`hemm` requires complex types. `--conjugate true` selects conjugated complex
dot products; the default is unconjugated. Index reductions retain the cuBLAS
one-based result convention and check ties against the first matching index.
For complex `asum`/index reductions, the reference uses the BLAS sum of absolute
real and imaginary components, rather than the complex Euclidean magnitude.

Vector experiments accept `--elements` (or `--n` as an alias), positive `--incx`,
and positive `--incy`. If both length options are supplied they must agree.
The result records `effective_elements` and the actual vector increments.
Matrix experiments use `--m`, `--n`, `--k`, `--ld-pad`, and applicable
transpose parameters. `syrk` uses N as the output order and K as the reduction
dimension. Triangular/symmetric operations accept `--side left|right` and
`--uplo lower|upper`. Triangular operations also accept `--diag unit|nonunit`.
`symm` and `hemm` require `--trans-a N`, since their APIs do not accept a
transpose argument.

For compatibility with saved case identities, older plans may carry dimensions
that do not participate in an operation: vector BLAS ignores m/k, SYRK ignores
m, and the other listed matrix BLAS operations ignore k. Explicitly supplied
inactive dimensions are named in `inactive_dimension_options`; they are not
reported as additional exercised dimensions.

Triangular inputs are well conditioned, with diagonal 2 (or 1 for unit diagonal)
and varying small off-diagonal values. The `trsm` check evaluates the residual
of the defining triangular equation using the actual solution. It does not
reuse a second GPU solve as a reference. Vector reductions check the complete
input, while matrix operations check deterministic output samples.

## Timing and correctness

- **Input setup is excluded.** Allocation, deterministic data generation, library
  handles/descriptors, and heuristic setup have separate setup metadata.
- **In-place inputs are restored.** Classic-cuBLAS GEMM with nonzero beta, AXPY, SCAL, SWAP,
  GEMV, SYRK, SYMM, HEMM, and TRSM restore their initial mutable input before each
  measured call. The restore is outside that call's CUDA event interval.
- **No growing recurrence is benchmarked accidentally.** Repeated nonzero-beta
  GEMM does not accumulate into C indefinitely. cuBLASLt and software GEMM keep
  C and D separate.
- **Reset timing is visible.** Such cases report
  `individual_cuda_events_excluding_input_reset` (or the analogous C-reset
  label). Resetting warms the restored input in cache; compare these cases
  under the same method on every GPU.
- **Sampling policy is shared.** Restored-input operations use the same bounded
  adaptive sample count and CV target as the common timer. `--repeats` fixes the
  number of calls per sample when positive and bypasses calibration. This timer
  rejects explicit repeats above 2,048 because it allocates a separate event
  pair per call; automatic repetition is capped there. Every individual event
  interval must be finite and positive. Warmup wall time includes input restores
  and stream completion; measured device time sums only operation intervals.
- **References use stored values.** FP16, BF16, FP8, and packed FP4 values are
  converted to their actual storage representation before CPU reference
  products. Complex arithmetic uses a long-double complex reference and the
  magnitude of the complex error.
- **Tolerances are reported.** The GEMM check records absolute/relative terms,
  input-conversion allowance, and an accumulation allowance scaled by the
  absolute product sum. This avoids treating cancellation-heavy outputs as a
  simple relative-error comparison.
- **Failures retain diagnostic evidence.** GEMM records `validation_failed_samples`
  and at most two detailed sample records under `validation_first_failure.*`
  and `validation_worst_failure.*`. Each includes the batch, row, column, logical
  index, expected and observed complex components, CPU pre-activation value,
  absolute/relative error, allowed bound, and `bound_sum_abs` (the scaled absolute
  product sum plus `|beta C|` used by that bound). Values are stored as
  high-precision strings, including nonfinite
  values when present. The worst failure is the largest error divided by its
  allowed bound among failed samples; it need not have the largest absolute
  error. These diagnostics do not change the acceptance rule.
- **Algorithm identity survives a failed check.** cuBLASLt records candidate
  algorithm/configuration attributes and workspace before each validation,
  including for rejected candidates. It records the selected identity before
  independent final validation. Failures before performance measurement have no
  performance samples. A failure of the postmeasurement check retains its raw
  timing evidence with `validation_failed` status, so it cannot count as a
  validated performance result. Distinguishing
  an epilogue precision effect from a library or reference defect still requires
  checking the retained evidence; different outcomes alone do not identify the
  cause.
- **Useful work is counted.** Real/integer MACs count as two operations; complex
  MACs count as eight real operations. Scale conversion, padding, and internal
  emulation work are not added to the useful GEMM operation count.
- **Bytes are estimates.** The byte metric is useful tensor traffic, not a
  measured DRAM transaction count. It excludes cache effects, internal library
  scratch, and duplicated transactions. Block-scale storage is separately
  recorded.
- **Execution is classified conservatively.** `library_selected` means a
  library selected the implementation. Use the profiling tools for evidence
  of the actual instruction path. Software cases use `software_simt_gemm` or
  `software_blas_simt`; they do not establish an unavailable native datatype path.

## Direct worker examples

Adjust the binary suffix to `sm80`, `sm89`, or `sm120a` for the actual GPU.

```bash
# Strict FP32 vs an explicit TF32 request.
build/bin/archbench_sm120a --family gemm --backend cublas \
  --dtype f32 --compute pedantic --m 4096 --n 4096 --k 4096
build/bin/archbench_sm120a --family gemm --backend cublas \
  --dtype f32 --compute tf32 --m 4096 --n 4096 --k 4096

# INT8 -> INT32 cuBLASLt, TN layout, algorithm search.
build/bin/archbench_sm80 --family gemm --backend cublaslt \
  --dtype i8 --output-dtype i32 --trans-a T --trans-b N \
  --m 1024 --n 1024 --k 4096 --algorithms 32

# Actual pointer-array batched complex GEMM.
build/bin/archbench_sm89 --family gemm --backend cublas \
  --dtype c64 --m 64 --n 64 --k 64 --batch 32 --batch-mode pointer \
  --trans-a C --alpha 0.75 --alpha-imag 0.125 --beta 0.25

# Mixed FP8 inputs with a fused bias + GELU request.
build/bin/archbench_sm89 --family gemm --backend cublaslt \
  --dtype fp8_e4m3 --dtype-b fp8_e5m2 --output-dtype bf16 \
  --trans-a T --trans-b N --m 1024 --n 1024 --k 4096 \
  --scale-a 0.5 --scale-b 2 --epilogue bias_gelu

# Physically packed NVFP4 and varying hardware-layout block scales.
build/bin/archbench_sm120a --family gemm --backend cublaslt \
  --dtype fp4_e2m1 --output-dtype f32 --scale-mode block16 \
  --trans-a T --trans-b N --m 1024 --n 1024 --k 4096

# Matching software decoding/FP32 accumulation on an SM80 GPU.
build/bin/archbench_sm80 --family gemm --backend software \
  --dtype fp4_e2m1 --output-dtype f32 --scale-mode block16 \
  --trans-a T --trans-b N --m 1024 --n 1024 --k 4096

# FP64 emulation request; inspect the retained-mantissa indicator afterward.
build/bin/archbench_sm120a --family gemm --backend cublas \
  --dtype f64 --compute fixedpoint --emulation-strategy eager \
  --mantissa-control dynamic --workspace-mib 1024 \
  --m 4096 --n 4096 --k 4096

# Same FP64 numerical problem, without claiming the fixed-point algorithm.
build/bin/archbench_sm120a --family gemm --backend software \
  --dtype f64 --compute fixedpoint --software-policy mathematical-reference \
  --m 4096 --n 4096 --k 4096

# Full-output throughput check and the linked example's sample/repeat counts.
build/bin/archbench_sm89 --family gemm --backend cublas \
  --dtype bf16 --output-dtype f32 --input-pattern rank1 \
  --cublas-algo default_tensor_op \
  --m 4096 --n 4096 --k 4096 --warmup 10 --warmup-ms 0 \
  --samples 5 --max-samples 5 --target-cv 0 --repeats 20

# A right-side conjugate-transposed triangular solve.
build/bin/archbench_sm80 --family blas --op trsm --dtype c128 \
  --m 256 --n 512 --side right --uplo upper --trans-a C
```

Each example executes its explicitly selected backend. Unsupported library
algorithms, shape requirements, or device capabilities remain separate
outcomes from a matching software result. The requested `cublas-algo` enum is
recorded; equal shape/data/timing/API settings still do not assert that another
library version or GPU selected the same internal cuBLAS algorithm.

## Deliberate limits

This module does not implement all cuBLAS routines, arbitrary mixed complex
storage, grouped variable-shape cuBLASLt APIs, every proprietary algorithm,
FP4/FP8 output quantization, all scale layouts, or all integer GEMM widths.
It implements the operations enumerated above. Named model quantization methods
such as GPTQ and AWQ are not library datatypes. They require an explicit packing
format and compatible kernel; this module does not rename a generic INT4 test
to imply such compatibility.

The authoritative API reference is
[NVIDIA cuBLAS 13.5 documentation](https://docs.nvidia.com/cuda/cublas/index.html).

BLAS also accepts `backend=native` as an alias for the cuBLAS API path; its
execution label remains library-selected, not a claim about native Tensor
Core instructions.
