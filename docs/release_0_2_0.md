# Version 0.2.0: audit findings and implemented changes

This release extends the supplied `archbench.zip` and its `reports/A.zip`
evidence. It retains the user's two compilation fixes and adds software
counterparts, stricter validation, robust tuning, build-level instruction
inspection, expanded workloads and complete automatic reports.

## What was unusual in the supplied results

The four directories contain **768 records: four runs of the same 192-case
acceptance matrix**. They are two physical GPU models at two power settings,
not four distinct GPU architectures. No SM89 run was supplied.

| Historical run | Passed measured numerical cases | Unsupported | Validation failures | Measurement only | Missing |
|---|---:|---:|---:|---:|---:|
| CMP 170HX, 150 W | 181 | 8 | 1 | 2 | 0 |
| CMP 170HX, 300 W | 181 | 8 | 1 | 2 | 0 |
| RTX PRO 6000 Blackwell, 300 W | 185 | 5 | 0 | 2 | 0 |
| RTX PRO 6000 Blackwell, 600 W | 184 | 5 | 1 | 2 | 0 |

The historical `profile=standard` field was the runner's default CLI argument;
it did not describe the active acceptance matrix. New runs record the effective
selection source, its SHA-256 and software-pair/shard policy explicitly.

### Zeros were not zero-duration kernels

The reports contain **739 timed medians and 5,173 raw samples**. Every recorded
timing sample and median is strictly positive. The smallest samples are about
1.44–2.36 microseconds across the four runs. The misleading zeros had several
other causes:

* A byte model is absent for many register microbenchmarks. A zero-initialized
  byte counter was being rendered as zero bandwidth rather than N/A.
* Transfers and launch/atomic/value operations were presented under a generic
  operation-rate label even when their useful unit was different.
* Unsupported or unvalidated cases inherited zero error fields even though no
  numerical error had been measured.
* Quantization streaming calculated absolute error but left relative error at
  its default zero. One u4/group32/4099 case had absolute error
  `2.384185791015625e-7` with a misleading relative zero.
* An actual very small rate in billions of launches per second can round to
  `0.000` under a fixed-decimal presentation. It should have a suitable unit or
  scientific notation, alongside latency in microseconds.
* A genuine exact-zero error is valid and is preserved as zero.

The report now carries explicit metric availability and domains. It uses
microseconds as the primary time unit, adds appropriate G/T work units, and
shows N/A for missing/unmeasured quantities. No historical measurement was
changed to manufacture a nonzero value.

### The BF16 bias/GELU result is a real validation failure

Case `gemm.d34ff5e502baa9293b57` is a 128³ cuBLASLt BF16 GEMM with
`alpha=.75`, `beta=.25` and `bias_gelu`. Three runs failed with maximum absolute
error `0.004580600481269786` and relative error `0.09649373786982883`.
The RTX 300 W run passed with different error values and a different selected
heuristic. A heuristic index is local to a returned candidate list; matching
indices on different GPUs do not identify the same implementation.

The old tuner selected the fastest candidate before checking its numerical
result. That allowed timing noise or a different selected implementation to
change whether the whole case passed. The new tuner restores input, validates
every returned candidate, records each candidate's configuration/errors/status,
and selects the fastest **validated** candidate by its median tuning time.
Rejected candidates stay in diagnostics. If none passes, the case fails and its
software counterpart remains a separate case. Tolerances were not increased
to hide the observed failure.

The old failing records also lost algorithm identity because it was written
only after successful validation. Algorithm and execution provenance now precede
validation, with bounded first/worst mismatch examples.

A host rounding study reproduced the passing error envelope with final BF16
rounding. An intermediate-rounding model reproduced one failing relative
statistic but not the observed absolute maximum. That is not sufficient to
identify the GPU algorithm's internal rounding sequence or blame the power
limit. The exact cause of those historical candidate outputs still requires a
new hardware run with the improved candidate diagnostics.

### Some skips were implementation gates

The old native tensor implementation gated FP64 MMA to SM80 and disabled
INT4/binary MMA on SM120a. The new build compiles the exact optional forms for
each target and records their code generation when binary utilities are
available. Under CUDA 13.5.37:

| Requested probe | SM80 machine code | SM89 machine code | SM120a machine code |
|---|---|---|---|
| FP64 8×8×4 MMA | DMMA | DMMA | DMMA |
| INT4 MMA | INT4 IMMA | INT4 IMMA | INT8 IMMA plus unpacking: compiler expansion |
| Binary MMA | BMMA | BMMA | INT8 IMMA plus conversion: compiler expansion |

These are static build observations. They do not prove that an untested board
executes a given instruction, its performance, or the exact dynamic path of a
library call. The worker retains runtime validation. An unsupported request,
compiler expansion, runtime error and verified numerical result remain distinct.

SM80's FP8/narrow requests still cannot gain the corresponding physical native
instructions from a scalar kernel. They now have matching, automatically
selected software cases. The old two SM80 cuBLASLt low-precision rejections
also gain software numerical counterparts. Library `NOT_SUPPORTED` or an empty
heuristic list does not make the underlying matrix product impossible.

### Other code defects found during the extension

* Deterministic stride sampling could repeatedly land on accumulator chain zero
  or one packed component. Sampling now stratifies lanes and includes every
  chain/component for each selected lane.
* Nonfinite values could fail a comparison while `max(0, NaN)` left error
  metrics at zero. Error accounting now preserves nonfinite failures as null
  summaries with explicit diagnostics; relevant downloaded outputs are scanned.
* Converting very large integer outputs to `double` before subtracting can hide
  adjacent-u64/u128 differences. Integer mismatch diagnostics now preserve
  encoded-bit-pattern differences.
* Some kernels choose blocks proportional to SM count. The same case ID then
  has different total work on different GPUs. The comparator now uses useful
  throughput for these cases and only uses latency ratios for equal work/bytes.
* Pointer-chase launch metadata and shared-memory working-set metadata did not
  describe the actual kernel. They now report the effective launch/allocation.
* Large reports could be limited by row rendering or excessive payload size.
  Every case is retained; large reports/comparisons use bounded offline pages.

## Software implementations added

| Family | Numerical counterpart |
|---|---|
| Tensor instruction tests | Existing uniform-fragment scalar equivalent, now automatically paired with each selected request; same types/scales/accumulator/pattern/control parameters |
| Dense GEMM | CUDA SIMT tiled multiply for supported real/integer/complex storage, mixed precision, transposes/conjugation, padding, batches, alpha/beta and epilogues |
| BLAS | CUDA kernels for all 16 implemented vector/matrix operations, sharing the native path's inputs and CPU references |
| FFT | CUDA radix-2 transforms per power-of-two axis and labeled direct DFT fallback for other lengths; C2C/R2C/C2R, ranks 1–3, batch, direction and normalization |
| Sparse | CSR SpMV/SpMM in FP32/FP64 with the same CSR and dense tensors |
| Quantized GEMM | SIMT decoded-weight multiply preserving each library path's encode/decode/timing stages; fused on-the-fly decode was already software |

These are measured on the selected GPU. Software FFT uses FP32 intermediate
arithmetic for FP16/BF16 storage; this difference is explicit. Reduction order
is generally different from a library. For cuBLAS fixed-point/BF16x9/fast-accum
requests, the counterpart is a mathematical reference, not an implementation
of the library's private algorithm. `pairs.json` and each detail view preserve
that relation and never turn a software pass into a native/library pass.

This release does not implement every external quantization ecosystem, all
cuDNN/NCCL operations, every CUDA instruction, arbitrary user-defined formats,
or optimized production kernels for every case. The scope inventory and
extension adapters remain explicit. All built-in GEMM requests have an admitted
software storage/policy route by source/configuration audit; actual execution
and resources still determine their outcomes.

## Timing and reporting

New default stable policy: 11–31 retained samples, 50 ms target batches,
10 warmups plus 200 ms warmup wall time, CV target 3%, and a bounded repeat cap.
Fixed-repeat and fixed-sample policies remain available. Unstable samples remain
visible. The report separates supported implementation, numerical validation,
measurement quality and experiment coverage.

Reports include microseconds/nanoseconds/milliseconds; domain-specific G/T
rates; GB/s/GiB/s where defined; spread, drift and median uncertainty; phase
costs; validation errors; algorithm/candidate details; requested/software links;
and complete missing/skipped/failing case accounting. These are derived views
of retained raw evidence, not replacement measurements.

See [README](../README.md) for full-run commands and
[methodology](methodology.md) for exact definitions and limitations.

## Reviewable source diff

`reports/source_review/source_changes_from_upload.patch` contains the complete
source/test/configuration/documentation diff against the uploaded project,
retaining both compilation fixes. Its application to a fresh copy of the
uploaded source was verified byte-for-byte. `diff_inventory.json` records every
changed path and before/after SHA-256. Build evidence and preserved historical
reports are included separately in the full archive rather than inflated into
that source patch.
