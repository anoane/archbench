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
