# Comparison with the two linked GPU benchmarks

The following source files were fetched and read on 25 November 2026. The
implementation here is independent; their source is not redistributed.

| Source | SHA-256 of inspected file |
|---|---|
| [FP8 MMA probe](https://github.com/JustAResearcher/gpu-unlock-ada/blob/main/docs/evidence/rtx4090-48g-modded/fp8-mma-bench.cu) | `d9dcc8c02625e27a54b78d91e7a85070f47761d98affb648374839900c21da86` |
| [GEMM benchmark](https://github.com/JustAResearcher/gpu-unlock-ada/blob/main/docs/evidence/rtx4090-pr1-stock/gemm_bench.cu) | `c35ec706f9587b9749bf369a35d4dfca9e88c666c4582726490e0138cdd6e240` |

## FP8 MMA

The linked program requests
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, with
1,056 blocks, 256 threads, eight independent accumulator chains and 20,000
iterations by default. It uses uniform E4M3 ones (`0x38`) and counts 8,192
floating operations per warp instruction. Five single-call samples are taken
and the best rate is reported. Its standalone probe prints four results and
checks one of them.

ArchBench already had this PTX arithmetic operation. The supplied 192-case
acceptance run did not use the same sustained-throughput controls. The new
`configs/tensor_throughput.json` includes the exact type/shape/grid/chain/loop
configuration and a paired scalar implementation, plus 69 further pairs over
other types, grids, accumulator and scale settings. It records median time/rate,
variation, every sample and all accumulator validation outputs. Half-accumulator
long loops can reset accumulation in bounded segments; reset overhead is timed
but does not inflate useful operation counts.

The harness is not byte-for-byte identical: ArchBench records clocks and more
output evidence, and stable mode uses warmup/adaptive batch timing. If you want
the linked single-call repetition policy, select the matching explicit case and
use `--samples 5 --max-samples 5 --target-cv 0 --warmup 0 --warmup-ms 0 --repeats 1`.
That sampling choice does not make the measured kernel free of instrumentation.

## FP16/BF16 GEMM to FP32

The linked program requests `cublasGemmEx` with NN layouts, FP16 or BF16 A/B,
FP32 output, `CUBLAS_COMPUTE_32F`, alpha one and beta zero, at
1024³, 4096³ and 8192³. It uses `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, ten warmup
calls, five timed batches of twenty calls, and reports median and best TFLOP/s.

Its structured inputs depend only on A's row and B's column:

* A(row, p) = `(row % 13 - 6) / 8`
* B(p, column) = `(column % 11 - 5) / 8`
* C(row, column) = K × A(row, 0) × B(0, column)

The binary-fraction values permit an exact analytic reference at these K values.
ArchBench's new `input-pattern=rank1` fills the same stored values and validates
every output, with bounded copies and no cubic-time CPU reference. The six
cases in `configs/gemm_throughput.json` request this pattern and the upstream
algorithm enum; automatic expansion adds six measured software cases. The
original random/seeded cases retain their existing independent sampled
references and remain useful for more varied numerical tests.

To request the linked sampling controls:

```sh
./bench run --matrix configs/gemm_throughput.json \
  --binary build/bin/archbench_sm89 --device 0 \
  --output runs/gemm_upstream_controls --timeout 600 \
  --quality standard --samples 5 --max-samples 5 --target-cv 0 \
  --warmup 10 --warmup-ms 0 --repeats 20
```

Add `--no-software-pairs` only when you deliberately want just the six library
cases. Preserve normal stable mode for broader timing diagnostics. Actual
cuBLAS algorithm selection still depends on the library version, GPU, storage
layout and environment. Matching API arguments does not establish identical
machine-code paths across devices or library versions.
