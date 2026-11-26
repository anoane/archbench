# Plans, measurements, and report formats

## Stable case and run identity

A case contains `family`, scalar `params`, `provider`, and optional `tags`.
Its ID is the family plus 20 hexadecimal characters from the SHA-256 of
canonical family/parameter/provider JSON. Tags do not alter identity. Saved
readers recheck the complete case content and hash; records outside the saved
plan, from another run, or with inconsistent case content are rejected.

Architecture, GPU, binary, timing policy, and environment belong to the run.
This permits identical requests to be joined across devices, while requiring
matching measurement and software provenance for ordinary comparisons. The
same case can still select a hardware-dependent launch grid; reported work
counts must therefore be checked before interpreting a latency ratio.

Resolved `samples`, `max-samples`, `target-cv`, `warmup`, `warmup-ms`, `target-ms`,
`repeats`, and `max-repeats` are part of policy identity. Preset names are a
convenience; actual numbers describe the measurement. Memory/device controls,
selection fingerprint, relevant environment, software fingerprints, timeout,
and shuffle order also constrain resume. See [methodology.md](methodology.md).

## Run directory

| Artifact | Meaning |
| --- | --- |
| `run.json` | Run identity, resolved policy, device/software/environment and selection provenance. |
| `plan.jsonl` | Immutable selected cases, written before execution. |
| `results.jsonl` | Append-only worker records, including attempts and process metadata. |
| `logs/` | Raw stdout/stderr for each attempted invocation. |
| `telemetry.jsonl` | Optional timestamped whole-worker NVML observations, not kernel-aligned measurements. |
| `coverage.json` | Derived selected/recorded/missing/validated and family/outcome counts. Reports also add quality, implementation, and pairing counts. |
| `pairs.json` | Requested/software counterpart relationships and separate numerical-pair coverage. |
| `results.csv` | Derived scalar statistics, semantic units, errors, availability, and quality for every selected case. |
| `report.html` | Self-contained detail report for small plans; offline aggregate index for large plans. |
| `report_index.json`, `report_details/` | Large-report routing manifest and bounded self-contained HTML detail files. Custom report filenames use the corresponding stem. |

Reports are generated views. They do not alter raw records or upgrade old
results to a new implementation. A legacy report can derive pairing from its
saved plan and correct a misleading display label while keeping original
execution/error fields inspectable. Old `results.jsonl` remains unchanged.

The runner locks its output against concurrent writers. An incomplete final
JSONL line may be repaired after interruption; complete corrupt records are
not silently discarded. Retries append attempts. Summaries use the latest
valid saved attempt for each case. An unrecorded planned case appears as a
report-only `missing` placeholder, not a fabricated worker terminal record.

## Worker record

The raw worker schema remains `schema_version=1`:

| Field | Interpretation |
| --- | --- |
| `status` | Execution outcome; separate from correctness, implementation, and sample quality. |
| `validation` | `passed`, `failed`, `not_run`, `not_checked`, or `not_applicable`; explicit passed variants in `model.py` are also accepted. |
| `execution` | Requested PTX, compiler expansion, software arithmetic, library selection, runtime API, or other path description. |
| `unit` | Backend's primary useful-rate unit; it does not make every work counter floating-point arithmetic. |
| `work_per_iteration` | Useful operations/items/calls per callback under its stated model. A zero means no useful-work rate is defined. |
| `bytes_per_iteration` | Logical useful bytes per callback. A zero means no byte model is provided, not that hardware moved no bytes. |
| `samples_ms` | Retained positive finite per-call timing samples, or an empty array when no final timing was obtained. |
| `metadata` | String-valued provenance, semantics, requested options, validation/timing scope, and code-generation evidence. |
| `metrics` | Numeric summaries, counters, timings, storage costs, error diagnostics, and observations. |
| `max_abs_error`, `max_rel_error` | Error maxima within the checked scope where actually computed. |
| `reason` | Unsupported, validation, resource, execution, or other explanation. |
| `case`, `run_id`, `attempt`, `process` | Runner binding to the immutable plan, attempt history, raw logs, and whole-process timing. |
| `pairing` | Runner-derived association with explicit requested/software numerical counterparts. |

Nonfinite numeric summaries serialize as JSON null. Every timing sample must
be finite and strictly positive; booleans are not timings. `ok` needs accepted
numerical validation and positive samples. Empty/graph launch measurements use
`measurement_only` with `validation=not_applicable` and never count as a
numerically validated measurement.

## Timing fields and phases

| Metadata / metrics | Scope |
| --- | --- |
| `sampling_method` | For example `cuda_event_batch`, `individual_cuda_events_excluding_reset`, or `individual_host_completed_calls_excluding_preparation`. |
| `timing_sample_scope` | Exact averaging and inclusion boundary for one saved sample. |
| `repetition_policy` | `fixed` or `automatic_target_duration`. |
| `sampling_stop_reason` | CV met, maximum samples reached, fixed count, insufficient samples, or a backend-specific fixed collection. |
| `timing_minimum_samples`, `timing_maximum_samples`, `timing_target_cv` | Requested sample bounds and dispersion threshold; zero CV disables extension. |
| `timing_sample_count`, `timing_repeats`, `timing_measured_calls` | Retained sample count, calls averaged per sample, and their product. |
| `timing_requested_fixed_repeats` | Zero requests calibration; positive values request that explicit repeat count. |
| `warmup_calls`, `warmup_host_ms`, `warmup_scope` | Actual warmup work and host duration through its stated synchronization. |
| `calibration_calls`, `calibration_host_ms`, `calibration_device_ms` | Calibration outside the final sample distribution. A fixed-repeat run performs no calibration calls. |
| `measurement_host_ms`, `measured_device_ms` | Host collection duration versus summed device operation intervals; these overlapping scopes must not be added as exclusive costs. |
| `*_preparation_host_ms`, `*_timed_calls_host_ms` | Host-transfer preparation and timed completed-call totals, with CPU migration/preparation kept outside operation intervals. |

Only applicable fields are present. The restored-input sampler excludes each
reset from device operation intervals but includes it in host wall time. Managed
migration restores CPU ownership before every measured call. Main launch samples
use events, while separate host submission/completion statistics have their own
scope. No phase metric is a substitute for aligned energy or DRAM counters.

All collected measurement samples remain in the distribution. Warmup and
calibration are separate phases. Extra sampling is bounded and stops on a CV
threshold or the configured cap; it does not guarantee future stability.

## Derived statistics and semantic rates

HTML detail data contains `stats`, `quality`, `rates`, `capability`, and
`implementation` in addition to original counters and metadata. CSV exposes
scalar columns; raw samples remain available in `results.jsonl` and the HTML.

| Derived fields | Definition |
| --- | --- |
| `median_ms`, `median_us`, `median_ns` | Median of retained samples; display uses microseconds consistently. |
| `mean_ms`, `min_ms`, `max_ms`, `stddev_ms`, `mad_ms` | Mean/extrema, sample standard deviation, median absolute deviation; microsecond variants are also exported. |
| `p10_batch_ms`, `p90_batch_ms` | Linearly interpolated sample quantiles; generally per-call batch averages rather than single-launch tails. |
| `cv` | Sample standard deviation / mean; unavailable with one sample. |
| `half_run_drift_fraction` | Last-half minus first-half median, divided by overall median; unavailable below four samples. |
| `median_ci95_low_ms`, `median_ci95_high_ms` | Central median interval endpoints, blank when no finite interval is available. |
| `median_ci_method`, `median_ci_target_coverage`, `median_ci_achieved_coverage`, `median_ci_finite` | Construction and nominal fixed-count coverage, with explicit finite-interval availability. |
| `useful_gops`, `useful_rate_unit`, `work_domain` | Legacy numeric giga-work rate plus its actual semantic unit/domain. This field is not universally GFLOP/s. |
| `useful_tera_rate`, `useful_tera_rate_unit` | The same useful count scaled to tera units. |
| `tflops`, `integer_tops` | Populated only for corresponding floating-point or integer arithmetic domains. |
| `logical_gbytes_s`, `logical_gibytes_s` | Logical payload/traffic model divided by median time, in decimal GB/s and binary GiB/s. |
| `useful_work_per_ns`, `ns_per_useful_work` | Throughput and its reciprocal; not instruction latency. |
| `useful_work_per_logical_byte` | Ratio of the two useful counters when both are positive. |
| `estimated_fft_tflops` | Conventional FFT arithmetic estimate; separate from actual transform-point work and software algorithm cost. |
| `*_ci_low`, `*_ci_high` | Rate bounds obtained by reciprocating time bounds in reverse order. |

Per-sample rate values and their extrema/mean/dispersion are available in HTML
detail data. Rate point estimates use count divided by median time, which can
differ from the median of reciprocals for an even number of samples.

Default intervals use binomial order statistics, not bootstrap resampling.
Fewer than six samples cannot give a finite distribution-free 95% median
interval. Achieved coverage is for fixed-count independent identically
distributed samples. CV-based adaptive stopping, drift, and correlation are
not corrected. See the explicit assumption/coverage-scope text in `stats`.

Undefined rates, unchecked errors, uncomputed legacy quantization relative
errors, and absent measurements display N/A and export blank CSV fields.
A genuine computed error of zero remains numeric zero. Tiny positive rates
retain significant digits instead of becoming misleading fixed-width zeros.
Legacy raw defaults are preserved separately for inspection.

Quality labels are `stable`, `variable`, `insufficient_samples`, `not_requested`,
and `not_measured`. They describe retained within-run sample averages and do not
change numerical validation or support outcomes. `quality_threshold_source`
distinguishes an actual run/worker policy from a post-hoc 0.05 report threshold
used to classify legacy records; it never implies adaptive collection occurred.
BLAS `asum`, `nrm2`, `iamax`, and `iamin` use `reduction_work` with
Gop/s/Top/s reduction-model units; their mixed abs/comparison/reduction models
are not reported as strict TFLOP/s, even for floating operands.

## Capability, code generation, and pairs

Capability includes `availability`, `scope`, and the original reason. Backend
requests, architecture checks, installation, build, resources, and unimplemented
functionality are distinct. Unavailable does not mean globally impossible.

Implementation classification distinguishes requested native PTX, compiler
lowering, promoted arithmetic, explicit software, compiler emulation, library
selection, runtime operations, and no execution. `compiler_expanded_tensor_ptx`
is classified as compiler emulation even when wider hardware tensor operations
are used internally. `tensor_codegen_evidence=direct_mma_sass` establishes instruction
selection in the standalone compiler probe, not runtime correctness or the
complete benchmark kernel instruction count. Full metadata remains visible.

`pairing` includes operation ID, role, requested backend, software case ID,
selected peers, relation, and explicit absence of bitwise/library-algorithm
identity claims. `pairs.json` counts requested implementations, selected and
unselected software counterparts, both/requested-only/software-only/neither
validated, and requested pairs with a validated implementation. Multiple requested
backends can share one mathematical operation; these counts are per request, not
a distinct-operation denominator. `unique_operation_counts` separately groups
operation IDs and reports selected, both/requested-only/software-only/neither
validated, and numerically covered operations. Relations
are `same_request` or `mathematical_reference`; see methodology for their limits.

Small reports enrich each pair with measured rates, quality, and a labeled
software/requested ratio where eligible. Large reports keep global pair JSON
compact and enrich only the pairs relevant to an opened detail shard.
Cross-shard counterpart links navigate to each original case.

## Outcomes and finite coverage

| Status | Numerical measured pass? | Interpretation |
| --- | --- | --- |
| `ok` + accepted validation + positive samples | Yes | Checked within recorded scope. |
| `measurement_only` | No | Valid nonnumerical timing. |
| `unsupported`, `not_implemented` | No | Unavailable backend/request; reason retained. |
| `architecture_mismatch` | No | Worker does not match the selected architecture. |
| `memory_budget`, `out_of_memory` | No | Planned budget or actual allocation failure. |
| `validation_failed`, `not_validated` | No | Numerical evidence failed or was absent. |
| `missing_dependency`, `missing_binary` | No | Required component absent. |
| `timeout`, `interrupted`, `error`, `compile_failed` | No | Failed or incomplete execution. |
| `probe_only` | No | Build/capability evidence without a measured numerical result. |
| Report-only `missing` | No | Selected case has no saved worker outcome. |

Requested, recorded, missing, numerically validated, measurement-only, and pair
coverage have different denominators. All requests recorded can mean complete
outcome accounting despite unsupported cases. It does not imply complete CUDA
feature coverage. Adapter/probe evidence never silently inflates core passes.

## Comparison and large-report layout

Exact-ID comparisons require compatible policy and software identity, subject
to the explicit software-mismatch override. A speedup is eligible only when both
records are numerically validated measurements or explicit measurement-only
launches. Missing/unsupported/failed cases have no ratio, not zero performance.

| `comparison_basis.basis` | Ratio |
| --- | --- |
| `equal_work_latency` | Baseline median / candidate median, with equal modeled useful work and bytes. |
| `useful_work_rate` | Candidate useful work/time divided by baseline useful work/time, for compatible work domains. |
| `logical_bandwidth` | Candidate modeled bytes/time divided by baseline modeled bytes/time when useful work is zero. |
| `unavailable` | No eligible measurements or incompatible/absent work model. |

Counts and timing scopes are retained alongside ratios. Bounds formed from
separate timing intervals are endpoint comparisons, not jointly calibrated
95% ratio confidence intervals. Backend identity and quality remain visible;
there is no aggregate score or average of unrelated units.

At most 5,000 cases use one report/comparison HTML. Above that, the top HTML is
an offline index with a small family/hash-prefix routing manifest. Each detail
HTML contains at most 1,000 cases and has its own pagination, filters and detail
view. Exact case-ID lookup routes without embedding one index entry per case.
Page-group filtering covers family/type/operation/outcome; full parameter search
is within the opened group. Every selected case remains reachable.

Large `comparison.json` is a manifest with `layout=offline_html_shards`; detailed
rows are in the listed HTML files and sibling JSON files under
`comparison_details/`. Small `comparison.json` directly contains `rows`.
`comparison.csv` is complete in both layouts. Keep the detail folder with the
index when copying reports. Raw evidence remains the authoritative record;
HTML/CSV/derived JSON can be regenerated without running a GPU benchmark.
