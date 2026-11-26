# Measurement and comparison methodology

ArchBench measures explicitly configured implementations on real CUDA GPUs.
Architecture target, storage format, accumulation policy, operation, and backend
are separate experimental choices. A result describes one request and machine;
it does not establish identical performance on every GPU with that capability.
Host tests, compiler acceptance, SASS inspection, numerical GPU validation, and
measured GPU performance are different evidence. See [validation.md](validation.md)
for the actual verification performed on this release.

## Experiment identity and numerical counterparts

A case ID hashes family, explicit parameters, and provider. Architecture, device,
measurement policy, and software environment belong to run identity. Native,
library, and software backends have distinct case IDs; a software result never
replaces a rejected requested implementation or changes its outcome.

`pairing` metadata associates explicit counterparts through an operation ID.
`same_request` preserves requested numerical controls; `mathematical_reference`
preserves the mathematical problem but does not reproduce a particular internal
library emulation strategy or fast accumulation schedule. Neither relation
asserts bitwise equality, algorithm equality, or identical timing scope. Pair
counts are per requested implementation; `unique_operation_counts` separately
counts distinct operation IDs so shared counterparts cannot inflate that denominator.
Software emulation runs arithmetic on the current GPU. It does not simulate
another architecture's execution time or physical interconnect bandwidth.

The immutable plan is the finite coverage denominator, including failed,
unsupported, timed-out, and unrecorded cases. The largest generated profile is a
finite matrix, not every CUDA operation, input bit pattern, shape, or algorithm.
Pair coverage separately answers whether either implementation produced a
validated numerical measurement; it does not establish support for a failed API.

## Timing policy

Most backends use one nonblocking stream and `context::measure`. One callback
can contain a kernel, several kernels, or several CUDA API launches. CLI presets
resolve to explicit policy values saved in run identity:

| `--quality` | Minimum / maximum samples | Minimum warmup calls / host duration | Target time per sample | Target CV |
| --- | --- | --- | --- | --- |
| `stable` (CLI default) | 11 / 31 | 10 / 200 ms | 50 ms | 0.03 |
| `standard` | 7 / 21 | 3 / 0 ms | 20 ms | 0.05 |
| `quick` | 3 / 3 | 1 / 0 ms | 5 ms | Disabled |

Explicit timing flags override preset values. Direct worker defaults are seven
minimum / 21 maximum samples, three warmup calls, zero minimum warmup duration,
20 ms target, CV 0.05, and 10,000 maximum repetitions. `--repeats 0` calibrates;
a positive value fixes calls per sample and must not exceed `--max-repeats`.

1. Complete the minimum warmup calls and synchronize; continue until elapsed
   warmup host time also meets `--warmup-ms`. These are two minimum conditions,
   not a promise of a particular thermal state or an extra duration afterward.
2. For automatic repeats, time one callback with CUDA events. Choose a bounded
   target/calibration ratio. If more than one repeat is selected, recalibrate
   with up to 16 calls. Fixed repeats bypass these calibration executions.
3. Place events around repeated callbacks and divide elapsed time by repeats.
   Retain every collected measurement sample without outlier rejection.
4. Collect at least `--samples`. If CV is enabled, continue until the retained
   samples' sample standard deviation / mean meets the target, or until
   `--max-samples`. A maximum reached with excessive CV remains visibly variable.

`samples_ms` normally contains **per-call averages of batches**, not individual
launch durations. A stream can wait for the CPU to enqueue callbacks, so event
intervals can include enqueue gaps. The target duration is a calibration goal:
repeat caps, clock changes, and input-dependent execution can change actual time.
The policy does not manufacture stability by clipping or selecting fast samples.

Warmup, calibration, and measurement have separate host duration/call-count
metrics. CUDA event paths additionally report calibration and measured device
interval totals. Calibration is outside the final sample distribution. Host
phase durations include their stated synchronization and bookkeeping; they are
not exclusive GPU execution measurements and should not be summed with device
durations as though the scopes were disjoint.

## Statistics and uncertainty

Reports expose minimum, median, mean, maximum, interpolated p10/p90, sample
standard deviation, CV, median absolute deviation, and first/last-half median
drift. Times are shown consistently in microseconds, with exact millisecond
samples and additional microsecond/nanosecond CSV columns. p10/p90 describe
retained per-call averages; they are not individual-launch tail latency.

The default median interval uses central binomial order statistics, with its
actual fixed-count coverage recorded. At least six independent samples are
needed for a finite distribution-free 95% interval. Smaller samples display no
finite interval, even if their observed values happen to be identical. A single
sample has no estimated standard deviation or CV. An explicit Python API
bootstrap option exists, but automatic reports do not run expensive resampling.

**Coverage is nominal for a fixed number of independent, identically distributed
samples.** CV-dependent early stopping is not corrected. Neither correlation,
thermal drift, nor between-run variation is removed. Thus the recorded
fixed-count coverage is not a guaranteed coverage for an adaptively stopped
experiment. First/last-half drift is descriptive evidence, not a calibrated test.

`stable`, `variable`, `insufficient_samples`, `not_requested`, and `not_measured`
are measurement-quality labels. They are separate from numerical validation and
capability. A wrong computation can have stable timings; a correct one can vary.
Repeat entire campaigns to assess repeat-run variation. Legacy records without
a CV policy use an explicitly labeled post-hoc report threshold; this does not
imply that their original collection adapted to CV.

Rates use count divided by median time. For an even sample count this need not
equal the median of per-sample reciprocals. Reports retain per-sample rates and
invert timing interval endpoints to obtain rate bounds. Endpoints combined from
two separate median intervals are not a jointly calibrated 95% speedup interval.

## Timing scopes

| Path | Included work and exclusions |
| --- | --- |
| Ordinary kernels and libraries | Events around callbacks; allocation, host inputs, and CPU reference work are outside. |
| Restored mutable BLAS inputs | Reset before each call's start event; sum individual operation intervals, excluding resets. Automatic repeats cap at 2,048; larger fixed counts are rejected. Warmup wall time includes resets. |
| Pinned H2D/D2H, D2D, available P2P | Repeated transfer callbacks timed with events; setup/pinned allocation excluded. |
| Pageable H2D/D2H | Average of individually synchronized host-clock calls, including pageable staging and completion. |
| Managed prefetch / first touch | Average of individually synchronized host-clock calls after CPU migration and full CPU write before **every** call. Preparation is excluded from operation time and reported separately. |
| Empty / graph launch | Main device-event samples plus separate host submission and synchronized-completion statistics. Host samples do not control the device CV stopping decision. |
| Native cuFFT C2R | Input-restoration D2D copy plus inverse FFT, because the library may overwrite its input. |
| Software FFT C2R | Immutable half-spectrum reconstruction, axis transforms, and packing; read the different preprocessing scope before interpreting a library/software ratio. |

Host transfer paths honor minimum/maximum samples, CV, warmup duration, and fixed
or calibrated repetitions. A host sample averages completed calls; its scope is
not identical to a single event interval surrounding an enqueued batch.
Managed preparation runs between repeats, preserving migration rather than
silently reusing resident pages. `timing_sample_scope`, `sampling_method`,
`timing_clock`, `timing_scope`, and preprocessing metadata describe the boundaries.

A launch callback contains `iterations` API calls. Each graph launch executes
`graph-nodes` empty kernels. Graph construction, instantiation, upload, and
synchronization are excluded and reported as setup. Main medians are per callback;
`host_ns_per_api_launch` divides host submission time by API calls, not graph nodes.

cuBLASLt selection times eligible returned heuristic candidates separately, then
validates and measures the selected candidate. This is not a search of all
internal algorithms. cuFFT planning and cuSPARSE preprocessing are outside their
execution callbacks. Quantized GEMM `fused`, `dequant_gemm`, `predequant_gemm`, and
`end_to_end` explicitly differ in inclusion of software encoding/decoding; even
`end_to_end` excludes allocation and host transfers. Inspect the recorded scope.

## Warm state, cache effects, and reset policy

Zero warmups does not establish a cold benchmark: input copies, calibration,
library searches, or correctness runs may already touch data. Most cases reuse
addresses without flushing cache. Allocation size or high logical bandwidth
does not prove L1/L2 residency or DRAM traffic. Restoring mutable inputs can warm
cache even when reset time is excluded.

Atomic state persists between timed repetitions and is reset for a fresh-state
validation launch. CAS comparing zero with an already changed value measures
steady-state failed comparisons. Low-precision unit increments can stop changing
the stored value. These are stated workloads, not arbitrary atomic update traces.

Pointer chasing repeatedly starts at one node in a random Hamiltonian cycle.
A bounded traversal may visit only a prefix of the allocation; visited nodes and
actual launch geometry are reported. Its `clock64` metric is a separate sample
with loop overhead. ALU cycle metrics likewise include scheduling/loop effects
and are not isolated SASS instruction-latency measurements.

Strided access `(i * stride) % elements` revisits a subset when stride and length
share factors. Shared-bank working-set bytes describe the per-block shared
allocation; distinct lane addresses avoid broadcast. Barrier/shared workloads
use launch geometry, rather than the arbitrary global element count, for work.
Default blocks can depend on SM count. Identical case IDs can therefore do
different amounts of work; the comparator explicitly accounts for that fact.

## Numerical correctness

A measured numerical pass needs accepted validation and positive finite samples.
Validation is generally separate from timing; it does not recheck every timed
repetition. Read reference descriptions, tolerances, and checked-element counts.
A pass establishes only the stated deterministic input and validation scope.

| Path | Reference / typical scope |
| --- | --- |
| Scalar/packed/integer/bit ALU | Deterministic CPU recurrence; generally up to 256 output registers. |
| Tensor microbenchmarks | Uniform fragment recurrence; narrower than randomized layout/GEMM conformance. |
| Conversions / integer-width arithmetic | Every generated output versus the stated host reference. |
| GEMM / matrix BLAS | Arithmetic from actual stored operands; typically sampled outputs, residuals for triangular solves. |
| Vector reductions | Full CPU reduction of generated input. |
| Memory, collectives, atomics, row workloads, transfers, CSR | All generated outputs; atomics use fresh-state validation. |
| Attention | Default 16 evenly spaced output rows, all channels, independent CPU attention. |
| Native/software FFT | Default 32 output elements versus independent long-double DFT; inverse is unnormalized. |
| Quantization | Packed bytes/scales/decoded values against portable codec, with independent host golden checks. |
| Quantized GEMM | Up to 257 outputs against long-double products of decoded weights. |

Sampling can miss other errors. `--validation-samples`, `--check-rows`, and
`--check-bins` expand applicable checks. Shared CPU/GPU codec logic alone cannot
establish specification conformance. Quantization distortion from original data
is different from implementation error against the decoded reference.
Maximum errors describe only checked outputs. Exact zero remains zero; undefined
or uncomputed error displays N/A. Nonfinite systems validation values fail and
serialize unavailable error maxima as null, with an explicit diagnostic count.

Compute Sanitizer/profiling outputs are separate evidence. Instrumentation
measurements do not replace ordinary timings. A checksum, successful launch,
or accepted capability probe is not numerical correctness evidence.

## Units and implementation evidence

For median time `t` milliseconds, useful giga-rate is `work / (t * 1e6)`;
tera-rate is `work / (t * 1e9)`. Floating arithmetic uses GFLOP/s and TFLOP/s;
integer arithmetic uses GOP/s and TOP/s. SFU evaluations, conversions, atomic
calls, loads, items, API launches, barriers, and bit-pairs retain distinct units.
BLAS absolute sums, norms, and index extrema are reduction-work models combining
abs/comparison/reduction work; `asum`, `nrm2`, `iamax`, and `iamin` do not receive
a strict TFLOP/s label.
A source FMA generally counts two useful operations; complex GEMM MACs count
eight. Internal emulation, unpacking, or scratch work does not inflate useful MACs.

Logical GB/s is `bytes / (t * 1e6)`; GiB/s is `bytes / (t * .001 * 2**30)`.
These model useful traffic, not measured DRAM transactions, cache misses, or
library scratch accesses. Transfers count one-way payload; atomics provide no
byte model. FFT point rates and a conventional FFT FLOP estimate are separate.
Work/ns and ns/work are throughput transforms, not instruction latency.
A zero/absent work or byte model yields N/A for that rate dimension.

Execution labels distinguish requested PTX, compiler/intrinsic lowering,
software arithmetic, compiler expansion, runtime APIs, and library selection.
`compiler_expanded_tensor_ptx` means the requested format expanded to other
operations, potentially using wider tensor instructions. `direct_mma_sass`
is standalone compiler-probe instruction evidence, not a full-kernel instruction
count; runtime capability/correctness still require
successful execution. A library API accepting a request does not establish its
chosen hardware path. Fixed-point-use metadata refers to the observed query/call.

Unavailable results retain scope: backend request, architecture, resources,
installation, build, or unimplemented functionality. An unavailable case does
not prove that its mathematical operation is globally impossible.
`measurement_only` launch cases have no numerical output and no numerical pass.

## Telemetry, provenance, and comparison

Optional NVML telemetry is tagged to the active worker and spans initialization,
setup, timing, validation, and possible inter-worker gaps. It is not aligned to
kernel event intervals. Sampled power or energy differences are not kernel energy
measurements; short operations may finish between samples. Unsupported queries
remain null. P2P accessibility alone does not identify NVLink versus PCIe.

Record power policy, clocks, cooling, CPU/NUMA placement, shared-system activity,
and topology. The harness does not lock those conditions. Run independent GPUs
or machines with matching saved plans/software and avoid shared-resource
contention. Repeat campaigns for between-run evidence.

Each case runs in an isolated process; timeout covers the entire worker including
setup and CPU validation. Resume checks GPU UUID, binary/selection hashes,
resolved measurement policy, environment, shuffle seed, timeout, and software
provenance. Completed results are append-only; only an incomplete final JSONL
line can be repaired. Reports use the latest attempt. Retry preserves history.

Exact-case comparisons check policy/environment and source/build/compiler/
runtime/driver/library identity. `--allow-software-mismatch` exposes an explicit
software mismatch; it does not waive policy checks. With equal modeled useful
work and bytes, speedup is baseline median / candidate median. Otherwise a
compatible positive useful-work model compares work/time; zero-work transfer
models can compare logical bytes/time. Missing/incompatible models yield no
ratio. The basis and counts remain visible. Failed numerical results are
ineligible; explicit measurement-only launches remain separately comparable.

Reports retain every selected case, including unrecorded placeholders. Small
runs have one self-contained HTML report. Above 5,000 cases, an offline index
links self-contained detail shards of at most 1,000 cases; exact-ID routing and
cross-shard counterpart links avoid embedding the whole dataset in the index.
CSV covers the complete plan. Large comparisons similarly use bounded HTML/JSON
shards and a complete CSV. Search within a shard examines its case details;
index filtering searches family/type/operation/outcome groups. Keep detail
folders with their index when moving reports. No overall GPU score is computed.

## Accounting and references

Requested, recorded, numerically validated, measurement-only, unavailable,
failed, and missing counts answer different questions. Pair coverage is another
explicit denominator. Complete outcome accounting can include unsupported
requests and does not establish complete CUDA feature or numerical coverage.
External adapters/probes retain their separate evidence and never inflate core
numerical passes. See [result-format.md](result-format.md) for fields/statuses.

Authoritative implementation: [timing](../src/common.cu), [systems](systems.md),
[ALU](alu.md), [GEMM](gemm.md), [statistics](../python/archbench/statistics.py),
[reporting](../python/archbench/report.py), [pairing](../python/archbench/pairing.py),
[runner identity](../python/archbench/runner.py), and [telemetry](../python/archbench/telemetry.py).
