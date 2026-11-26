# Validation evidence for version 0.2.0

**Checked 25 November 2026. Toolkit: CUDA 13.5.0, NVCC 13.5.37.**

This release has an actual CUDA build and host regression evidence. It does
**not** have new GPU numerical/performance or Compute Sanitizer results from
the authoring workspace. The supplied historical results are preserved under
`reports/historical_A`; they describe the earlier user-built source revision.

## Completed checks

| Check | Observed result | What it establishes |
|---|---|---|
| Complete CUDA workers | SM80, SM89 and SM120a compiled and linked successfully | All shipped CUDA translation units and enabled library calls passed the real CUDA 13.5 compiler/linker for these targets |
| Enabled libraries | cuBLAS 13.10.0.20, cuFFT 12.5.0.14, cuSPARSE 12.9.4.2 from NVIDIA's CUDA 13.5 redistributable manifest | Actual headers/libraries were used, not stub APIs |
| CMake/CTest | 3/3 registered test groups passed | Release-build codec checks, limb reference program and the full Python host suite run through the build system |
| Final Python host suite | **132 tests passed** | Arithmetic helpers, independent references, candidate filtering, interval calculations, semantic metrics, pairing/selection/UUID controls, reporting, sharding, provenance and process handling |
| Portable codecs | **393,666 checks passed** | Host encoding/decoding, packing, group/storage boundaries and specified numerical policies |
| Two-limb integers | **532,480 transitions passed** | Signed/unsigned widths 1–128 against independent `unsigned __int128` arithmetic |
| Software FFT/model | **96 transforms; 30,407 checks** | Shared arithmetic/indexing against independent multidimensional DFTs, Hermitian layouts, butterfly partitions and native input contracts |
| Software BLAS | Dense-algebra and triangular-equation residual checks passed | Shared structured/triangular operations across sides, triangles, transposes, conjugation, padding and strides |
| Software GEMM/model | Encoding, rounding, packed/batched layouts, scales, rank-one sums and tuner selection tests passed | Host-visible parts of the actual software implementation, including invalid-candidate exclusion |
| Quantized decoded GEMM | 245 dense products passed, including poisoned padding and active nonfinite input checks | Correct ragged-K indexing and use of active decoded weights |
| Sampling diagnostics | 50 lane/chain/packing shapes plus NaN/Inf and adjacent-u64/u128 cases passed | The previous stride bias and false-zero mismatch diagnostics are addressed |
| Optional instruction probes | 9 target/form combinations compiled and disassembled | Static acceptance and direct-MMA versus compiler-expanded classification |
| Bounded full-kernel SASS audit | 9 selected kernel checks passed | Retained ILP accumulator chains and relevant machine-code paths in these specific compiled kernels |
| Full catalogue | **404,129** distinct selected cases | Current generator/configuration/pair expansion count, not executions or passes |
| Historical report regeneration | **1,564** raw files verified unchanged | Original run/plan/results/logs/telemetry preserved while derived views were regenerated |

Some checks are also exercised inside the 132-test suite. Their counts describe
different scopes, not additional independent GPU executions. CPU tests use the
shared portable code where possible and independent algebra/encoding references.
They are not CUDA device execution and cannot certify CUDA synchronization,
register layouts, dynamic library behavior, or device memory safety.

The toolchain was installed locally from NVIDIA's official manifest. Component
SHA-256 values were verified before use. GNU 14.2.0, CMake 4.4.5 and Ninja were
used. The downloaded SDK, build objects and executables are not distributed in
this source archive. Rebuild on the benchmark machine to obtain local binaries.

## Recorded evidence

| Path | Contents |
|---|---|
| `validation/build_provenance.json` | Compiler/library/component identities, SDK hashes, completed commands and output binary hashes |
| `validation/configure.log.gz`, `build.log.gz` | Actual final configure, compiler/assembler and link output |
| `validation/ctest.log` | Final CTest run, including all 132 Python tests and the two C++/reference test groups |
| `validation/host_tests.log` | Earlier direct 131-test pass before the last reduction-unit regression was added |
| `validation/capability_probes/` | Nine probe logs, SASS and machine-readable code-generation classifications |
| `validation/kernel_audit/` | Exact extraction commands, object/source hashes and nine bounded full-kernel SASS checks |
| `validation/worker_sm*_help.log` | Successful executable startup and help output without creating a GPU context |
| `validation/gpu_initialization_attempt.jsonl` | Actual failed initialization in this workspace, retained as unavailability evidence |
| `validation/full_selection.json` | CLI count of the complete expanded catalogue |
| `validation/input_audit/` | Original source diff, raw report audit and BF16 rounding investigation |
| `reports/historical_A/report_regeneration_audit.json` | Raw-file preservation and regenerated report checks |
| `validation/historical_v0_1_0/` | Unmodified earlier release's authoring evidence; not evidence for new code |

The final GPU initialization attempt returned a CUDA driver/runtime availability
error before any benchmark kernel ran. Therefore no new TFLOP/s, microsecond,
GPU numerical, or GPU sanitizer results are claimed here. The successful
compiler checks do not override that limit.

## Static code-generation findings

CUDA 13.5.37 accepted FP64 MMA, INT4 MMA and binary MMA probe forms for all three
targets. FP64 probes emitted DMMA instructions; INT4/binary emitted direct
IMMA/BMMA on SM80/SM89. On SM120a the INT4/binary requests expanded into INT8
IMMA with conversion/unpacking logic. Report labels preserve that distinction.

The bounded full-kernel inspection found eight independent HMMA/QMMA chains in
the selected native tensor kernels, eight FFMA/HFMA2 chains in scalar/packed
kernels, and 32 FP32 accumulator registers in the selected scalar tensor
baseline. Expanded SM120a subbyte operations call shared helper bodies eight
times per loop; static helper opcode counts are not dynamic instruction counts.

This audit is source-specific and bounded. It is not a verifier for every
kernel, nor evidence of runtime support, silicon execution resources, dynamic
instruction counts or measured peak throughput. cuBLAS-selected paths remain
opaque unless separately profiled or disassembled with their actual kernels.

## Reproduce and complete the remaining checks

```sh
./bench selftest --verbose
./bench build --arch 80,89,120a \
  --nvcc /usr/local/cuda-13.5/bin/nvcc --jobs 4
ctest --test-dir build --output-on-failure

python3 tools/audit_alu_sass.py --build build \
  --cuobjdump /usr/local/cuda-13.5/bin/cuobjdump

python3 tools/validate_gpu.py --arch 120a \
  --gpu-uuid GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --output runs/acceptance_sm120a
```

Repeat acceptance on the SM80 and SM89 GPUs with their matching workers. The
acceptance matrix's 192 historical requests expand to 235 distinct cases with
software counterparts. The script checks numerical outcomes and sanitizer
representatives; one representative does not prove an entire family safe.
Inspect unsupported, failed and unavailable cases as well as the exit code.

Remaining gates are real-device numerical execution of the new revision,
Compute Sanitizer, runtime library availability and candidate behavior,
per-kernel profiler counters, repeatable hardware measurements under recorded
conditions, and Docker/CI execution. The Dockerfile and CI workflow are included
but were not run here. Generated HTML JavaScript syntax, report contents,
filters' data and shard routing are tested; a full visual browser inspection
was unavailable in this environment.
