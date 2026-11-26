// SPDX-License-Identifier: AGPL-3.0-or-later
#include "archbench/common.cuh"
#include "archbench/codec.hpp"
#include "archbench/quant_gemm.hpp"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <numeric>

namespace archbench {
namespace {
using codec::format;
using codec::scale_kind;
constexpr unsigned quant_threads = 128;

__global__ void encode_groups(const float* source, std::size_t n, std::size_t group,
                             format f, unsigned char* packed, unsigned char* scales,
                             float* offsets, scale_kind scale_type) {
  for (std::size_t g = blockIdx.x * blockDim.x + threadIdx.x; g < (n + group - 1) / group;
       g += std::size_t(blockDim.x) * gridDim.x) {
    const auto count = n - g * group < group ? n - g * group : group;
    codec::encode_group(source + g * group, count, f,
                        packed + g * codec::packed_bytes(group, f.bits), scales, g, scale_type, offsets);
  }
}

__device__ float decoded(const unsigned char* packed, const unsigned char* scales,
                         const float* offsets, std::size_t index, std::size_t group,
                         format f, scale_kind scale_type) {
  const auto g = index / group;
  const auto code = codec::unpack(packed + g * codec::packed_bytes(group, f.bits), index % group, f.bits);
  return codec::decode(code, f) * codec::load_scale(scales, g, scale_type) + offsets[g];
}

__global__ void decode_values(const unsigned char* packed, const unsigned char* scales,
                             const float* offsets, float* output, std::size_t n,
                             std::size_t group, format f, scale_kind scale_type) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += std::size_t(blockDim.x) * gridDim.x)
    output[i] = decoded(packed, scales, offsets, i, group, f, scale_type);
}

__global__ void weight_only_gemm(const float* a, const unsigned char* weights,
          const unsigned char* scales, const float* offsets, float* c,
          int m, int n, int k, int padded_k, std::size_t group, format f, scale_kind scale_type) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || column >= n) return;
  float sum = 0;
  for (int p = 0; p < k; ++p)
    sum = fmaf(a[std::size_t(row) * k + p], decoded(weights, scales, offsets,
         std::size_t(column) * padded_k + p, group, f, scale_type), sum);
  c[std::size_t(row) * n + column] = sum;
}

__global__ void decoded_weight_gemm(const float* a,const float* b,float* c,
        int m,int n,int k,int padded_k) {
  const int row=blockIdx.y*blockDim.y+threadIdx.y;
  const int column=blockIdx.x*blockDim.x+threadIdx.x;
  if(row<m&&column<n)c[std::size_t(row)*n+column]=quant_baseline::decoded_dot(a,b,row,column,k,padded_k);
}

struct host_quantized {
  std::vector<unsigned char> packed, scales;
  std::vector<float> offsets, decoded;
};

host_quantized reference_quantize(const std::vector<float>& values, std::size_t group,
                                 format f, scale_kind scale_type) {
  const auto groups = (values.size() + group - 1) / group;
  host_quantized ref;
  ref.packed.resize(checked_product(groups, codec::packed_bytes(group, f.bits)));
  ref.scales.resize(checked_product(groups, codec::scale_bytes(scale_type)));
  ref.offsets.resize(groups);
  ref.decoded.resize(values.size());
  for (std::size_t g = 0; g < groups; ++g) {
    const auto count = std::min(group, values.size() - g * group);
    auto* packed = ref.packed.data() + g * codec::packed_bytes(group, f.bits);
    codec::encode_group(values.data() + g * group, count, f, packed,
                        ref.scales.data(), g, scale_type, ref.offsets.data());
    const float scale = codec::load_scale(ref.scales.data(), g, scale_type);
    for (std::size_t i = 0; i < count; ++i)
      ref.decoded[g * group + i] = codec::decode(codec::unpack(packed, i, f.bits), f) * scale + ref.offsets[g];
  }
  return ref;
}

std::uint64_t random_word(std::uint64_t i) {
  i += 0x9e3779b97f4a7c15ull;
  i = (i ^ (i >> 30)) * 0xbf58476d1ce4e5b9ull;
  i = (i ^ (i >> 27)) * 0x94d049bb133111ebull;
  return i ^ (i >> 31);
}

std::vector<float> input_values(std::size_t n, const std::string& distribution) {
  std::vector<float> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto r = random_word(i + 19);
    float x = float(int(r & 65535) - 32768) / 8192.0f;
    if (distribution == "outliers" && i % 257 == 0) x *= 64;
    else if (distribution == "zeros") x = 0;
    else if (distribution == "constant") x = 0.375f;
    else if (distribution == "ramp") x = float(int(i % 1025) - 512) / 128;
    values[i] = x;
  }
  return values;
}

void quant_metadata(result& r, format f, std::size_t n, std::size_t group,
                    scale_kind scale_type, std::size_t stored_bytes) {
  r.execution = "software_emulated_numeric_format";
  r.metadata["packing"] = "little-bit-first; group-aligned; not a GGUF/GPTQ/AWQ/bitsandbytes file layout";
  r.metadata["scale_storage"] = scale_type == scale_kind::f32 ? "fp32" : scale_type == scale_kind::e8m0 ? "ue8m0 upward power-of-two" : "e4m3 RNE";
  r.metadata["offset_storage"] = "fp32 per group; zero for symmetric formats";
  r.metadata["encoder"] = "reference implementation; one thread per group; not a peak-throughput quantizer";
  r.metadata["nan_policy"] = "float NaNs preserved where format permits; finite-only formats map NaN to zero";
  r.metadata["rounding"] = "nearest, ties even; saturate finite; unsigned affine / signed symmetric";
  r.metadata["validation_reference"] = "host portable codec; independently cross-checked by Python golden tests";
  r.metrics["bits_per_value_payload"] = f.bits;
  r.metrics["physical_bytes"] = double(stored_bytes);
  r.metrics["effective_bits_per_value"] = n ? double(stored_bytes) * 8 / n : 0;
  r.metrics["group_size"] = double(group);
}

bool quant_stream(context& ctx, result& r) {
  const auto f = codec::parse(ctx.opts.get("format", "s4"));
  const auto scale_type = codec::parse_scale(ctx.opts.get("scale-format", "f32"));
  const auto n_arg = ctx.opts.integer("elements", 1048576), group_arg = ctx.opts.integer("group", 32);
  if (n_arg <= 0 || n_arg > (1ll << 32) || group_arg < 1 || group_arg > 65536)
    throw std::invalid_argument("quant elements must be 1..2^32 and group 1..65536");
  const auto n = std::size_t(n_arg), group = std::size_t(group_arg), groups = (n + group - 1) / group;
  const auto packed_n = checked_product(groups, codec::packed_bytes(group, f.bits));
  const auto scale_n = checked_product(groups, codec::scale_bytes(scale_type));
  const auto physical = packed_n + scale_n + groups * sizeof(float);
  const auto op = ctx.opts.get("op", "roundtrip");
  if(op!="encode"&&op!="decode"&&op!="roundtrip") { skip(r,"unknown quant operation");return true; }
  quant_metadata(r,f,n,group,scale_type,physical);
  r.unit="Gvalue/s";
  r.work_per_iteration=double(n);
  r.bytes_per_iteration=(op=="roundtrip"?2.0:1.0)*(double(n)*sizeof(float)+physical);
  ctx.memory_guard(checked_product(n, 2 * sizeof(float)) + physical);
  const auto distribution = ctx.opts.get("distribution", "uniform");
  if (distribution != "uniform" && distribution != "outliers" && distribution != "zeros" &&
      distribution != "constant" && distribution != "ramp") throw std::invalid_argument("unknown distribution");
  const auto host = input_values(n, distribution);
  const auto ref = reference_quantize(host, group, f, scale_type);
  device_buffer<float> source(n), output(n), offsets(groups);
  device_buffer<unsigned char> packed(packed_n), scales(scale_n);
  CUDA_CHECK(cudaMemcpyAsync(source.get(), host.data(), source.bytes(), cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemsetAsync(packed.get(), 0, packed.bytes(), ctx.stream));
  const unsigned grid = unsigned(std::min<std::size_t>(65535, (n + quant_threads - 1) / quant_threads));
  const unsigned encode_grid = unsigned(std::min<std::size_t>(65535, (groups + quant_threads - 1) / quant_threads));
  auto encode = [&] { encode_groups<<<encode_grid, quant_threads, 0, ctx.stream>>>(source.get(), n, group, f, packed.get(), scales.get(), offsets.get(), scale_type); };
  auto decode = [&] { decode_values<<<grid, quant_threads, 0, ctx.stream>>>(packed.get(), scales.get(), offsets.get(), output.get(), n, group, f, scale_type); };
  encode(); decode(); CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
  const auto actual_packed = read_buffer(packed), actual_scales = read_buffer(scales);
  const auto actual_offsets = read_buffer(offsets), actual = read_buffer(output);
  bool ok = actual_packed == ref.packed && actual_scales == ref.scales && actual_offsets == ref.offsets;
  long double mse = 0, signal = 0;
  double quant_max = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double error = std::abs(double(actual[i]) - ref.decoded[i]);
    // Preserve nonfinite mismatches as unavailable JSON metrics instead of
    // letting std::max silently retain zero when its second argument is NaN.
    const double reported_error = std::isfinite(error) ? error : std::numeric_limits<double>::infinity();
    r.max_abs_error = std::max(r.max_abs_error, reported_error);
    r.max_rel_error = std::max(r.max_rel_error,
        std::isfinite(error) ? error / std::max(1e-8, std::abs(double(ref.decoded[i]))) : reported_error);
    if (!std::isfinite(actual[i]) || error > 2e-6 * std::max(1.0, std::abs(double(ref.decoded[i])))) ok = false;
    const long double q_error = static_cast<long double>(ref.decoded[i]) - host[i];
    mse += q_error * q_error; signal += static_cast<long double>(host[i]) * host[i];
    quant_max = std::max(quant_max, double(std::abs(q_error)));
  }
  r.metadata["relative_error_definition"] = "max |actual-reference| / max(1e-8, |reference|)";
  r.validation = ok ? "passed" : "failed";
  if (!ok) { r.status = "validation_failed"; r.reason = "packed codes/scales/offsets or decoded values differ from host codec"; return true; }
  if (op == "encode") r.samples_ms = ctx.measure(encode);
  else if (op == "decode") r.samples_ms = ctx.measure(decode);
  else if (op == "roundtrip") r.samples_ms = ctx.measure([&] { encode(); decode(); });
  else { skip(r, "unknown quant operation"); return true; }
  r.metrics["quantization_rmse"] = std::sqrt(double(mse / n));
  r.metrics["quantization_max_abs_error"] = quant_max;
  r.metrics["relative_l2_quantization_error"] = signal ? std::sqrt(double(mse / signal)) : 0;
  r.metadata["validation_scope"] = "all packed bytes, all scales, all offsets and all decoded values";
  return true;
}

void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("quant GEMM cuBLAS status " + std::to_string(int(status)));
}

bool quant_gemm(context& ctx, result& r) {
  const auto f = codec::parse(ctx.opts.get("format", "s4"));
  const auto scale_type = codec::parse_scale(ctx.opts.get("scale-format", "f32"));
  const auto group_arg = ctx.opts.integer("group", 32);
  const auto m_arg = ctx.opts.integer("m", 16), n_arg = ctx.opts.integer("n", 1024), k_arg = ctx.opts.integer("k", 1024);
  if (m_arg < 1 || n_arg < 1 || k_arg < 1 || m_arg > 65536 || n_arg > 65536 || k_arg > 65536 || group_arg < 1 || group_arg > 65536)
    throw std::invalid_argument("quant GEMM dimensions/group must be 1..65536");
  const int m = int(m_arg), n = int(n_arg), k = int(k_arg), group = int(group_arg);
  const int padded_k = ((k + group - 1) / group) * group;
  const auto a_count = checked_product(m, k), b_count = checked_product(n, padded_k), c_count = checked_product(m, n);
  const auto groups = b_count / group;
  const auto packed_n = checked_product(groups, codec::packed_bytes(group, f.bits));
  const auto scale_n = checked_product(groups, codec::scale_bytes(scale_type));
  const auto physical = packed_n + scale_n + groups * sizeof(float);
  const auto mode=ctx.opts.get("mode","fused");
  if(mode!="fused"&&mode!="dequant_gemm"&&mode!="predequant_gemm"&&mode!="end_to_end") {
    skip(r,"unknown quant GEMM mode");return true;
  }
  const auto backend=ctx.opts.get("backend",mode=="fused"?"software":"cublas");
  if(backend!="software"&&backend!="emulated"&&backend!="cublas") {
    skip(r,"quant GEMM backend must be software/emulated or cublas");return true;
  }
  if(mode=="fused"&&backend=="cublas") {
    skip(r,"this fused packed-weight kernel is a software implementation; use a dequantized mode for cuBLAS");return true;
  }
  const bool use_library=backend=="cublas";
  quant_metadata(r,f,std::size_t(n)*k,group,scale_type,physical);
  r.execution=use_library?"software_dequantization_cublas_gemm":"software_dequantization_simt_gemm";
  r.unit="Gflop/s";
  r.metadata["work_domain"]="floating_point";
  r.metadata["requested_backend"]=backend;
  r.metadata["compute_implementation"]=use_library?"cuBLAS SGEMM after explicit software dequantization":"portable FP32 SIMT FMA dot products";
  r.metadata["cublas_used"]=use_library?"true":"false";
  ctx.memory_guard(checked_product(a_count + b_count * 2 + c_count, sizeof(float)) + physical);
  const auto distribution = ctx.opts.get("distribution", "uniform");
  if (distribution != "uniform" && distribution != "outliers" && distribution != "zeros" && distribution != "constant" && distribution != "ramp")
    throw std::invalid_argument("unknown distribution");
  auto host_a = input_values(a_count, "uniform"), host_b = input_values(b_count, distribution);
  const auto dtype = ctx.opts.get("dtype", "f32");
  if (dtype != "f16" && dtype != "bf16" && dtype != "f32") { skip(r, "quant GEMM activation dtype f16/bf16/f32 only"); return true; }
  for (auto& x : host_a) {
    x /= 4;
    if (dtype == "f16") x = __half2float(__float2half_rn(x));
    if (dtype == "bf16") x = __bfloat162float(__float2bfloat16_rn(x));
  }
  for (int column = 0; column < n; ++column)
    std::fill(host_b.begin() + std::size_t(column) * padded_k + k, host_b.begin() + std::size_t(column + 1) * padded_k, 0.0f);
  const auto ref = reference_quantize(host_b, group, f, scale_type);
  device_buffer<float> a(a_count), b(b_count), unpacked(b_count), c(c_count), offsets(groups);
  device_buffer<unsigned char> packed(packed_n), scales(scale_n);
  CUDA_CHECK(cudaMemcpyAsync(a.get(), host_a.data(), a.bytes(), cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(b.get(), host_b.data(), b.bytes(), cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemsetAsync(packed.get(), 0, packed.bytes(), ctx.stream));
  auto encode = [&] { encode_groups<<<unsigned(std::min<std::size_t>(65535, (groups + 127) / 128)), 128, 0, ctx.stream>>>(b.get(), b_count, group, f, packed.get(), scales.get(), offsets.get(), scale_type); };
  auto decode = [&] { decode_values<<<unsigned(std::min<std::size_t>(65535, (b_count + 127) / 128)), 128, 0, ctx.stream>>>(packed.get(), scales.get(), offsets.get(), unpacked.get(), b_count, group, f, scale_type); };
  struct blas_handle {
    cublasHandle_t value{};
    explicit blas_handle(bool create) { if(create)blas_check(cublasCreate(&value)); }
    ~blas_handle() { if (value) cublasDestroy(value); }
  } handle(use_library);
  if(use_library) {
    blas_check(cublasSetStream(handle.value, ctx.stream));
    blas_check(cublasSetMathMode(handle.value, CUBLAS_PEDANTIC_MATH));
  }
  const float alpha = 1, beta = 0;
  auto gemm = [&] {
    if(use_library)blas_check(cublasSgemm(handle.value,CUBLAS_OP_T,CUBLAS_OP_N,
                     n,m,k,&alpha,unpacked.get(),padded_k,a.get(),k,&beta,c.get(),n));
    else {
      decoded_weight_gemm<<<dim3((n+15)/16,(m+7)/8),dim3(16,8),0,ctx.stream>>>(a.get(),unpacked.get(),c.get(),m,n,k,padded_k);
      CUDA_CHECK(cudaGetLastError());
    }
  };
  auto fused = [&] { weight_only_gemm<<<dim3((n + 15) / 16, (m + 7) / 8), dim3(16, 8), 0, ctx.stream>>>(a.get(), packed.get(), scales.get(), offsets.get(), c.get(), m, n, k, padded_k, group, f, scale_type); };
  std::function<void()> launch;
  if (mode == "fused") launch = fused;
  else if (mode == "dequant_gemm") launch = [&] { decode(); gemm(); };
  else if (mode == "predequant_gemm") launch = gemm;
  else if (mode == "end_to_end") launch = [&] { encode(); decode(); gemm(); };
  else { skip(r, "unknown quant GEMM mode"); return true; }
  encode(); decode(); launch(); CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
  const auto actual = read_buffer(c);
  bool ok = true;
  const auto sample_count = std::min<std::size_t>(c_count, 257);
  for (std::size_t i = 0; i < sample_count; ++i) {
    const auto index = sample_count == c_count ? i : (i * (c_count - 1) / (sample_count - 1));
    const auto row = index / n, column = index % n;
    long double expected = 0, abs_sum = 0;
    for (int p = 0; p < k; ++p) {
      const long double term = static_cast<long double>(host_a[row * k + p]) * ref.decoded[column * padded_k + p];
      expected += term; abs_sum += std::abs(term);
    }
    const double error = std::abs(double(expected) - actual[index]);
    const double tolerance = 2e-5 + double(abs_sum) * 2e-6;
    const double reported_error=std::isfinite(error)?error:std::numeric_limits<double>::infinity();
    r.max_abs_error = std::max(r.max_abs_error, reported_error);
    r.max_rel_error = std::max(r.max_rel_error, std::isfinite(error)?error/std::max(1e-8,std::abs(double(expected))):reported_error);
    if (!std::isfinite(actual[index]) || error > tolerance) ok = false;
  }
  r.validation = ok ? "passed" : "failed";
  r.metadata["relative_error_definition"]="max |actual-reference| / max(1e-8, |reference|)";
  if (!ok) { r.status = "validation_failed"; r.reason = "quantized GEMM differs from long-double reference"; return true; }
  r.samples_ms = ctx.measure(launch);
  r.metadata["activation_storage"] = "FP32 storage of values rounded to requested activation dtype";
  r.metadata["output_accumulator"] = "FP32";
  r.metadata["gemm_math_mode"] = "CUBLAS_PEDANTIC_MATH or explicit FP32 FMA";
  r.metadata["validation_scope"] = "up to 257 stratified output entries; long-double dot products using decoded weights";
  r.metadata["native_low_precision_claim"] = "none: these are portable software baselines, separate from native tensor and cuBLASLt cases";
  r.metadata["preprocessing"] = mode == "end_to_end" ? "encoding+decode+GEMM timed" : mode == "dequant_gemm" ? "decode+GEMM timed; quantization excluded" : mode == "predequant_gemm" ? "GEMM only; both encode/decode excluded" : "on-the-fly dequantization+SIMT GEMM; initial quantization excluded";
  r.metadata["byte_metric"] = "ideal unique operand/output traffic; not total kernel load transactions";
  r.work_per_iteration = 2.0 * m * n * k;
  r.bytes_per_iteration = double(a.bytes() + c.bytes());
  if (mode == "fused") r.bytes_per_iteration += physical;
  else if (mode == "predequant_gemm") r.bytes_per_iteration += unpacked.bytes();
  else if (mode == "dequant_gemm") r.bytes_per_iteration += physical + 2.0 * unpacked.bytes();
  else r.bytes_per_iteration += b.bytes() + 2.0 * physical + 2.0 * unpacked.bytes();
  r.metadata["acceptance_threshold"] = "abs_error <= 2e-5 + 2e-6 * sum(abs(products)); engineering threshold, not a worst-case gamma_K bound";
  r.metrics["validated_output_entries"] = sample_count;
  long double mse = 0;
  for (std::size_t i = 0; i < b_count; ++i) { const long double e = static_cast<long double>(ref.decoded[i]) - host_b[i]; mse += e * e; }
  r.metrics["weight_quantization_rmse_including_padding"] = std::sqrt(double(mse / b_count));
  return true;
}

enum class width_op : int { add, sub, mul, bit_and, bit_or, bit_xor, shl, shr, minimum, maximum };
__device__ std::uint64_t width_compute(std::uint64_t a, std::uint64_t b, unsigned bits, bool sign, width_op op) {
  const auto mask = bits == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << bits) - 1;
  a &= mask; b &= mask;
  const auto sign_bit = std::uint64_t(1) << (bits - 1);
  const unsigned shift = unsigned(b % bits);
  std::uint64_t out = 0;
  switch (op) {
    case width_op::add: out = a + b; break;
    case width_op::sub: out = a - b; break;
    case width_op::mul: out = a * b; break;
    case width_op::bit_and: out = a & b; break;
    case width_op::bit_or: out = a | b; break;
    case width_op::bit_xor: out = a ^ b; break;
    case width_op::shl: out = a << shift; break;
    case width_op::shr: out = a >> shift; if (sign && (a & sign_bit) && shift) out |= mask ^ (mask >> shift); break;
    case width_op::minimum: out = (sign ? (a ^ sign_bit) < (b ^ sign_bit) : a < b) ? a : b; break;
    case width_op::maximum: out = (sign ? (a ^ sign_bit) > (b ^ sign_bit) : a > b) ? a : b; break;
  }
  return out & mask;
}
__global__ void width_kernel(const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* out,
                             std::size_t n, unsigned bits, bool sign, width_op op) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += std::size_t(blockDim.x) * gridDim.x)
    out[i] = width_compute(a[i], b[i], bits, sign, op);
}

bool integer_width(context& ctx, result& r) {
  const auto bits_arg = ctx.opts.integer("bits", 4), n_arg = ctx.opts.integer("elements", 1048576);
  if (bits_arg < 1 || bits_arg > 64 || n_arg < 1 || n_arg > (1ll << 32)) throw std::invalid_argument("integer width 1..64, elements 1..2^32 required");
  const unsigned bits = unsigned(bits_arg); const auto n = std::size_t(n_arg);
  const bool sign = ctx.opts.flag("signed");
  const std::map<std::string, width_op> operations = {{"add",width_op::add},{"sub",width_op::sub},{"mul",width_op::mul},
    {"and",width_op::bit_and},{"or",width_op::bit_or},{"xor",width_op::bit_xor},{"shl",width_op::shl},{"shr",width_op::shr},{"min",width_op::minimum},{"max",width_op::maximum}};
  const auto found = operations.find(ctx.opts.get("op", "add"));
  if (found == operations.end()) { skip(r, "integer-width operation unsupported"); return true; }
  const auto op = found->second;
  ctx.memory_guard(checked_product(n, 3 * sizeof(std::uint64_t)));
  std::vector<std::uint64_t> a(n), b(n);
  const auto mask = bits == 64 ? ~std::uint64_t(0) : (std::uint64_t(1) << bits) - 1;
  for (std::size_t i = 0; i < n; ++i) { a[i] = random_word(i) & mask; b[i] = random_word(i + n) & mask; }
  device_buffer<std::uint64_t> da(n), db(n), dc(n);
  CUDA_CHECK(cudaMemcpyAsync(da.get(), a.data(), da.bytes(), cudaMemcpyHostToDevice, ctx.stream));
  CUDA_CHECK(cudaMemcpyAsync(db.get(), b.data(), db.bytes(), cudaMemcpyHostToDevice, ctx.stream));
  auto launch = [&] { width_kernel<<<unsigned(std::min<std::size_t>(65535, (n + 255) / 256)),256,0,ctx.stream>>>(da.get(),db.get(),dc.get(),n,bits,sign,op); };
  launch(); CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
  auto output = read_buffer(dc);
  bool ok = true;
  for (std::size_t i = 0; i < n; ++i) {
    // Independent host reference uses a widened signed mathematical domain.
    const __int128 modulus = __int128(1) << bits;
    const __int128 signed_a = sign && (a[i] & (std::uint64_t(1) << (bits - 1))) ? __int128(a[i]) - modulus : __int128(a[i]);
    const __int128 signed_b = sign && (b[i] & (std::uint64_t(1) << (bits - 1))) ? __int128(b[i]) - modulus : __int128(b[i]);
    const auto shift = unsigned(b[i] % bits);
    std::uint64_t expected = 0;
    switch (op) {
      case width_op::add: expected = std::uint64_t(__uint128_t(a[i]) + b[i]); break;
      case width_op::sub: expected = a[i] - b[i]; break;
      case width_op::mul: expected = std::uint64_t(__uint128_t(a[i]) * b[i]); break;
      case width_op::bit_and: expected = a[i] & b[i]; break;
      case width_op::bit_or: expected = a[i] | b[i]; break;
      case width_op::bit_xor: expected = a[i] ^ b[i]; break;
      case width_op::shl: expected = std::uint64_t(__uint128_t(a[i]) << shift); break;
      case width_op::shr: expected = std::uint64_t(signed_a >> shift); break;
      case width_op::minimum: expected = std::uint64_t(signed_a < signed_b ? signed_a : signed_b); break;
      case width_op::maximum: expected = std::uint64_t(signed_a > signed_b ? signed_a : signed_b); break;
    }
    expected &= mask;
    if (output[i] != expected) { ok = false; break; }
  }
  r.validation = ok ? "passed" : "failed";
  if (!ok) { r.status = "validation_failed"; r.reason = "integer-width result differs from host 128-bit reference"; return true; }
  r.samples_ms = ctx.measure(launch);
  r.execution = "software_width_emulation";
  r.metadata["semantics"] = "two's complement modulo 2^bits; signed comparison/arithmetic right shift; shift count modulo width";
  r.metadata["storage"] = "uint64_t per value; not packed";
  r.metadata["validation_scope"] = "every output, host 128-bit reference";
  r.metadata["benchmark_kind"] = "streaming operation plus memory; use integer family for instruction dependency/ILP tests";
  r.work_per_iteration = double(n); r.bytes_per_iteration = double(n) * 3 * sizeof(std::uint64_t);
  return true;
}
}  // namespace

bool run_quant(context& ctx, result& r) {
  const auto family = ctx.opts.get("family");
  if (family == "quant") return quant_stream(ctx, r);
  if (family == "quant_gemm") return quant_gemm(ctx, r);
  if (family == "integer_width") return integer_width(ctx, r);
  return false;
}
}  // namespace archbench
