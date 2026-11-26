# Quantization, packing, and weight-only references

This module separates a numeric format, its physical representation, its scale
policy, quantization quality, and the execution path that consumes it. A
generic packed 4-bit value is not an implementation of every 4-bit model
quantization system.

## Supported numeric encodings

| CLI format | Payload | Value interpretation |
|---|---:|---|
| `s1` … `s16` | 1–16 bits | Two's-complement signed integer, symmetric group scaling |
| `u1` … `u16` | 1–16 bits | Unsigned integer, affine group scale and offset |
| `e2m1` | 4 bits | Finite FP4 encoding |
| `e2m3`, `e3m2` | 6 bits | Finite FP6 encodings |
| `e4m3`, `e5m2` | 8 bits | FP8 encodings with format-specific special values |
| `nf4` | 4 bits | The 16-entry NormalFloat4 codebook, not an exponent/mantissa format |
| `f16`, `bf16` | 16 bits | Binary16 and bfloat16 storage conversions, scale one |

The 40 names above are the portable codec's formats. This codec is shared by the
host reference and CUDA reference kernels; independent Python golden tests
check all small floating-point decode bit patterns, NF4, representative encode
values and tie cases. [Authoring validation](validation.md) states exactly what
was executed. Host validation does not establish CUDA device code correctness.

`s1` has the two's-complement integer values **{-1, 0}**. It is not a bipolar
{-1, +1} binary neural-network encoding. Binary warp MMA is a separate family.

## Layout and scaling

Values are packed low bit first, with each group starting at a byte boundary.
An incomplete group retains a full group-sized payload allocation and zeroed
unused payload bits. Six-bit formats really occupy a six-bit packed stream;
they are not silently allocated as one byte per value. Scales and offsets are
stored separately.

`--scale-format` controls actual scale storage:

| Value | Physical scale bytes/group | Policy |
|---|---:|---|
| `f32` | 4 | FP32 group scale |
| `e8m0` | 1 | Unsigned exponent-only power-of-two scale, rounded upward |
| `e4m3` | 1 | FP8 E4M3 scale rounded to nearest, ties even |

All groups additionally carry an FP32 offset. The offset is zero for symmetric
formats; it is used by unsigned affine formats. This deliberately simple layout
is documented and included in byte accounting. It is not an optimized file
container and is not ABI-compatible with vendor block-scale tensors, GGUF,
GPTQ, AWQ, bitsandbytes, or EXL2/EXL3.

Integer and small-float encoders use finite saturation and nearest-even code
selection. The group scale and affine offset are part of the operation being
benchmarked. A quantized scale may make values saturate; that distortion belongs
in the quantization quality metrics. FP16/BF16 codecs use scale one so ordinary
storage conversion is not confused with max-normalized quantization.

The code defines individual special-value encodings, and host tests cover
selected NaN/infinity/zero behavior. The end-to-end group-quantization benchmark
generates **finite** inputs. It does not constitute an exhaustive nonfinite,
subnormal, overflow, or stochastic-rounding conformance test.

## Streaming benchmark

```sh
build/bin/archbench_sm120a --family quant --format s3 --group 32 \
  --elements 65539 --op roundtrip --scale-format f32

build/bin/archbench_sm80 --family quant --format e2m1 --group 16 \
  --elements 65539 --op decode --scale-format e4m3
```

Options:

| Option | Choices / limits |
|---|---|
| `format` | The 40 codec names above |
| `op` | `encode`, `decode`, `roundtrip` |
| `elements` | 1 through 2^32, subject to device and host memory |
| `group` | 1 through 65,536 |
| `scale-format` | `f32`, `e8m0`, `e4m3` |
| `distribution` | `uniform`, `zeros`, `constant`, `ramp`, `outliers` |

The encoder is a correctness-oriented baseline: one CUDA thread processes a
group sequentially. It is intentionally identified as a reference encoder and
must not be described as the hardware's maximum quantization throughput.
The decoder uses a parallel value stream. Encode/decode setup needed to make
inputs valid is done before the timed region for single-stage measurements.

Before timing, every packed byte, scale byte, offset, and decoded value is
compared against the host codec. Decoded values use a small tolerance for host
and device floating-point differences. The record distinguishes:

- **Implementation error:** difference between CUDA output and the specified
  quantized reference. This controls pass/fail.
- **Quantization error:** RMSE, maximum absolute error, and relative L2 error
  between decoded quantized values and the original finite source. A low-bit
  format can pass implementation checks and still have substantial distortion.
- **Storage cost:** actual payload + scales + offsets, including ragged-group
  padding, with effective bits per source value.

Throughput is values per second. Reported logical bytes are minimum stage
traffic based on these buffers, not hardware transaction counts or measured
DRAM bandwidth. Repeated timing can use cache-resident buffers.

## Weight-only GEMM paths

`family=quant_gemm` uses packed weights in N rows of padded K values and computes
an M×N result from an M×K activation matrix. K padding ensures group boundaries
stay within a weight row. Padded values are zero and do not contribute to the
logical dot product.

```sh
build/bin/archbench_sm89 --family quant_gemm --format nf4 \
  --dtype bf16 --group 64 --m 16 --n 1024 --k 1024 --mode fused
```

| `mode` | Timed operations | Interpretation |
|---|---|---|
| `fused` | Unpack, load scale/offset, dequantize, scalar FP32 multiply-accumulate | On-the-fly SIMT reference baseline |
| `dequant_gemm` | Decode weights to FP32 + cuBLAS SGEMM | Dequantization overhead plus library GEMM |
| `predequant_gemm` | cuBLAS SGEMM only; decode performed before timing | Reused decoded weights |
| `end_to_end` | Encode weights + decode weights + cuBLAS SGEMM | Full reference conversion pipeline |

For the three decoded-weight modes, `backend=software` replaces SGEMM with a
real GPU SIMT FP32-FMA kernel. It preserves the same activation values, packed
weights, group scales/offsets, ragged-K padding, output reference, and timed
preprocessing stages. The runner selects this counterpart automatically beside
each cuBLAS request. `fused` is already a software implementation; requesting a
cuBLAS backend for that fused decode kernel is explicitly unavailable.
The software baseline reports useful `2MNK` FLOP counts and its own measured
time. It does not simulate a Tensor Core kernel's latency or claim optimized
weight-only inference throughput.

Both absolute and relative implementation errors are calculated. Nonfinite
errors serialize as JSON null and are accompanied by failed validation; an
unmeasured relative error is never substituted with a successful zero.

`dtype=f16`, `bf16`, or `f32` controls **activation value rounding**. In this
module activations are physically stored as FP32, and arithmetic/accumulation
is FP32. Thus an `s4` + `dtype=f16` case is not a native packed W4A16 kernel and
must not be compared as if it had FP16 activation memory traffic. The record
explicitly states this storage and execution policy.

For native-library low-precision GEMM requests, use `family=gemm` with
`backend=cublaslt`; it has physically packed FP4 inputs, FP8 inputs, and the
library-required scale layouts. Those have separate case identities and
capability checks. See [GEMM documentation](gemm.md).

Weight-only validation uses an independent long-double dot-product accumulation
over the **actual rounded activation values and decoded quantized weights**.
It checks up to 257 output positions, including boundary positions; it does not
check every output of a large GEMM. This validates the requested quantized
problem, not equivalence to the original unquantized weight matrix. FLOP rates
count useful 2MNK work; stage-specific byte rates are ideal minimum traffic,
not actual access multiplicity in the naive fused kernel.

## Relation to MXFP4, MXFP6, MXFP8, and NVFP4

Three different tests must stay distinct:

1. **Numeric emulation:** this portable codec with a selected payload, group
   size, and scale encoding. It tests numerical representation and baseline
   software cost.
2. **Warp instruction microbenchmark:** `family=tensor` selects a native
   architecture-specific MMA form, including supported mixed narrow formats
   and block scales. It uses register fragments and uniform structured data.
3. **Native library problem:** cuBLASLt/CUTLASS accepts real matrix layouts,
   scales, dimensions, and an available kernel. Performance includes the
   implementation and memory system of that kernel.

Selecting an E2M1 payload and E4M3 scale in the portable codec does not prove
that the GPU executed NVFP4 instructions. An FP4 library request does not prove
that every quantization producer's tensor layout can be consumed unchanged.
Use metadata, the API support result, kernel-specific profiling, and SASS
evidence to substantiate a path claim.

## Extending format support

New numeric encodings need a precisely defined code-to-value mapping, rounding,
saturation and special-value rules, physical packing, scales, offsets, and an
independent reference. New ecosystem formats additionally need their real
producer/runtime: tensor ordering, block metadata, permutations, calibration,
outlier rules, and any model-level transforms are part of the format.

Do not rename generic INT4 results to GPTQ/AWQ/GGUF/IQ/EXL results. The external
provider adapter can preserve runs from those implementations, while this
project's core matrix remains an auditable definition of the implemented cases.
The relevant official specifications and NF4 source are linked in
[sources](sources.md), with third-party notices in the project root.
