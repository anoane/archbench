# Software CUDA CSR and FFT baselines

`src/systems_software.cu` provides GPU implementations of the suite's CSR and
Fourier-transform workloads. Select `--backend software` or `--backend emulated`.
Both names select the same implementation. These kernels run on the selected
GPU and use the shared CUDA-event measurement policy; host reference execution
never contributes a GPU performance number.

The library-backed and software cases are separate results. A library request
that cannot run retains its unsupported/error status. A software result measures
the specified CUDA algorithm and does not establish the performance of an
unavailable native instruction or library algorithm.

## Supported operations

| Family | Operations | Stored data types | Controls |
| --- | --- | --- | --- |
| `sparse` | `csr_spmv`, `csr_spmm` | `f32`, `f64` | `rows`, `cols`, `nnz-per-row`; `n` for SpMM |
| `fft` | `c2c`, `r2c`, `c2r` | `f16`, `bf16`, `f32`/`c64`, `f64`/`c128` | `rank`, `n` or `nx`, `ny`, `nz`, `batch`, `direction`, `check-bins` |

Both families accept `threads` (default 256, supported multiple of 32) and
`blocks` (default four times the device's SM count, capped by the available
work). Other measurement controls come from the common harness. CSR offsets,
CSR column indices, FFT dimensions/batch, and per-transform FFT distance retain
the same positive int32 limits as the library baselines. Allocation sizes and
products are checked before allocation.

## CSR implementation and matching inputs

One CUDA thread computes each output element by traversing its complete CSR
row. SpMM uses column-major dense input and output matrices; SpMV is the
one-column case. Each product is accumulated using FP32 or FP64 FMA according
to the requested data type. The exact CSR column generation, per-row sorting,
stored values, and dense-input generator match the existing cuSPARSE cases.
Every output element is checked against an independent long-double host sum.

The reported work is `2 * nnz * dense_columns`, with unit `Gflop/s` and work
domain `floating_point`. Each multiply-add counts two floating operations.
Logical byte counts include the CSR arrays, dense input, and output once; they
do not describe physical DRAM traffic or the repeated reuse of dense values.

The software algorithm does not need a library preprocessing stage. A paired
case's `preprocess` control has no computational effect on this path, and the
result records that fact. Input generation and sorting are setup work. Timing
includes the GPU multiplication only. This is general CSR arithmetic; it does
not emulate a structured-sparse Tensor Core instruction or claim an optimized
SpMV/SpMM implementation.

## FFT algorithm, layout, and precision

Transforms are separable across the fastest `nx` axis, then `ny`, then `nz`.
Rank 1 uses `nx`; rank 2 uses `ny × nx`; rank 3 uses `nz × ny × nx`. Batches are
contiguous. Each power-of-two axis uses a bit-reversal kernel followed by
in-place radix-2 butterfly stages. Each other axis uses a direct DFT kernel.
Mixed shapes use both algorithms. The direct DFT fallback is quadratic in that
axis length and may be much slower than a library FFT for a large prime axis.
It is identified in the execution label and axis-count metrics.

`c2c` supports forward and inverse directions. `r2c` requires forward;
`c2r` requires inverse. The inverse is unnormalized: it is not divided by the
transform volume. Real-transform complex storage retains `nx/2 + 1` entries
along the fastest axis, including the correct multidimensional Hermitian
relationships on the boundary planes. C2R reconstructs the missing frequencies
on the GPU before applying the axis transforms.

The stored inputs match the library cases exactly, including salts, batch
offsets, compact Hermitian storage, and the `1/sqrt(volume)` input scale used
for FP16/BF16. FP16/BF16 inputs are decoded to FP32 complex arithmetic; outputs
are rounded to the requested storage type after all axes. FP32 uses FP32 complex
arithmetic and FP64 uses FP64 complex arithmetic. This is a storage-compatible
software reference algorithm. Its intermediate rounding does not reproduce a
library's implementation-specific sequence of narrow operations.

Unlike the native half/BF16 cuFFT path, the software path also accepts
non-power-of-two axes through the direct DFT fallback. That result remains a
software measurement; it does not turn a rejected library configuration into
a successful native result.

## Timing and work accounting

Each measured FFT invocation includes GPU input unpacking, Hermitian
reconstruction where needed, every axis kernel, and output conversion/packing.
The input stays immutable across repeats. Root tables, allocation, input
generation/uploads, and CPU validation are setup work outside timing. Setup
wall time is separately recorded as `algorithm_setup_ms`.

The native C2R cuFFT baseline includes an input-restoration D2D copy because
the library may overwrite its input. The software path instead reconstructs
its full complex workspace from immutable compact input every invocation. Both
measurement scopes are reported explicitly. Their elapsed times compare the
implemented end-to-end GPU invocation scopes, not identical kernel sequences.

FFT primary work is `batch * transform_volume`, reported as `Gitem/s`, with
work domain `item`. `estimated_fft_flops` is the conventional transform model,
kept separate from primary work. `software_algorithm_arithmetic_model` estimates
the actual chosen axis algorithm: `5 * total_points * log2(axis_length)` for
radix-2 axes, and `8 * total_points * axis_length` for direct complex DFT axes.
Neither estimate is a measured machine-instruction count. Unpacking, packing,
bit-reversal addressing, and launch overhead remain timed but are not included
in these arithmetic estimates.

Logical FFT byte counts cover stored input and output once. The extra full
complex workspace passes are excluded, so these byte counts must not be
interpreted as physical memory bandwidth. `temporary_storage_bytes` includes
both full complex workspaces and all axis root tables.

## Validation and acceptance checks

FFT validation uses an independent long-double multidimensional DFT, applied
to the actual stored input values. `--check-bins` defaults to 32 evenly spaced
output elements across all batches. Setting it to at least the output element
count checks all outputs. Both real and imaginary components are checked for
complex outputs. Each result records the sample count, tolerances, input scale,
validation scope, and error maxima. Nonfinite differences produce an infinite
error metric and a validation failure rather than a false zero.

The portable helpers in `include/archbench/systems_software_model.hpp` are used
by the actual GPU kernels and by `tests/systems_software_host.cpp`. The host test
checks 96 FP32/FP64 transforms against an independent DFT over 12 shapes, two
batches, both C2C directions, and real forward/inverse transforms. It covers
odd/even fastest dimensions, mixed FFT/DFT axes, lengths of one, complete
butterfly partitions, bit-reversal permutations, and compact Hermitian
reconstruction. It also compares the stored-input generator with the actual
pure host generator extracted from `src/systems.cu`.

```bash
PYTHONPATH=python python3 -m unittest discover -s tests -p test_systems_software.py -v
```

Those checks exercise arithmetic and layout code on a host compiler. CUDA
compilation, kernel launches, narrow conversion behavior, memory safety under
Compute Sanitizer, and performance require their separate on-device gates.
Useful small on-device canaries include all dtypes for `c2c` in both directions,
`r2c`, and `c2r`; ranks 1–3; batch 2; power-of-two shapes; and mixed shapes such
as `nx=5, ny=3, nz=2`. Set `check-bins` above each canary's output count so every
output is validated. CSR canaries should include `nnz-per-row=cols`, wrapped
column patterns, both storage types, and multiple SpMM columns.

For example, after building the matching architecture worker:

```bash
build/bin/archbench_sm89 --family fft --backend software --op c2r --dtype f64 \
  --rank 3 --nx 5 --ny 3 --nz 2 --batch 2 --check-bins 10000
build/bin/archbench_sm89 --family sparse --backend software --op csr_spmm \
  --dtype f32 --rows 17 --cols 23 --nnz-per-row 7 --n 5
```

Use the worker built for the selected GPU. A successful result requires actual
device execution and numerical validation; selecting an architecture name or
passing a host test alone does not establish that result.
