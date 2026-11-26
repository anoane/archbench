// SPDX-License-Identifier: AGPL-3.0-or-later
// CUDA arithmetic and warp-MMA microbenchmarks. See docs/alu.md for semantics.
#include "archbench/common.cuh"
#include "archbench/tensor_model.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace archbench {
namespace {

enum class int_op { add, sub, mul, mad, div, rem, min, max, bit_and, bit_or, bit_xor, shl, shr };
enum class float_op { add, sub, mul, fma, div, sqrt, min, max, abs };
enum class bit_op { popc, clz, brev, prmt, lop3, shf };
enum class sfu_op { rcp, rsqrt, sin, cos, exp2, log2 };

struct micro_config {
    int threads;
    int blocks;
    int iterations;
    int ilp;
    bool latency;
    size_t lanes() const { return size_t(threads) * size_t(blocks); }
};

micro_config configuration(const context& ctx) {
    const auto mode = ctx.opts.get("mode", "throughput");
    if (mode != "throughput" && mode != "latency")
        throw std::invalid_argument("mode must be throughput or latency");
    const auto bounded = [&](const char* key, int64_t fallback, int64_t low, int64_t high) {
        const int64_t value = ctx.opts.integer(key, fallback);
        if (value < low || value > high)
            throw std::invalid_argument(std::string(key) + " is outside the supported range");
        return int(value);
    };
    micro_config c;
    c.latency = mode == "latency";
    c.threads = bounded("threads", c.latency ? 32 : 256, 32, ctx.prop.maxThreadsPerBlock);
    const bool per_sm = !ctx.opts.get("blocks-per-sm").empty();
    if (per_sm && (!ctx.opts.get("blocks").empty() || c.latency))
        throw std::invalid_argument("blocks-per-sm is a throughput control and cannot be combined with blocks");
    const int blocks_per_sm = bounded("blocks-per-sm", 4, 1, 64);
    c.blocks = bounded("blocks", c.latency ? 1 : int64_t(ctx.prop.multiProcessorCount) * blocks_per_sm, 1, 65535);
    c.iterations = bounded("iterations", 512, 1, 1000000);
    c.ilp = bounded("ilp", c.latency ? 1 : 4, 1, 8);
    if (c.threads % 32)
        throw std::invalid_argument("microbenchmark threads must be a positive warp multiple within device limits");
    if (c.ilp != 1 && c.ilp != 2 && c.ilp != 4 && c.ilp != 8)
        throw std::invalid_argument("ilp must be 1, 2, 4, or 8");
    if (c.latency && (c.ilp != 1 || c.blocks != 1 || c.threads != 32))
        throw std::invalid_argument("latency mode requires one block, one warp, and ilp=1");
    return c;
}

template <typename T>
std::vector<T> copy_output(const device_buffer<T>& b, cudaStream_t stream) {
    std::vector<T> output(b.size());
    CUDA_CHECK(cudaMemcpyAsync(output.data(), b.get(), output.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return output;
}

std::vector<size_t> micro_sample_indices(size_t lanes, size_t values_per_lane) {
    // Sampling a flat array with a fixed stride can repeatedly select chain 0
    // or the low packed component. Select lanes, then include all their values.
    if (!lanes || !values_per_lane || values_per_lane > 256)
        throw std::invalid_argument("invalid microbenchmark validation shape");
    const size_t sampled_lanes = std::min(lanes, size_t(256) / values_per_lane);
    std::vector<size_t> indices;
    indices.reserve(sampled_lanes * values_per_lane);
    for (size_t sample = 0; sample < sampled_lanes; ++sample) {
        const size_t lane = sampled_lanes == 1 ? 0 : sample * (lanes - 1) / (sampled_lanes - 1);
        for (size_t value = 0; value < values_per_lane; ++value)
            indices.push_back(lane * values_per_lane + value);
    }
    return indices;
}

void record_micro(result& r, const micro_config& c, const device_buffer<unsigned long long>& clock,
                  cudaStream_t stream, double operations_per_step = 1) {
    r.work_per_iteration = double(c.lanes()) * c.ilp * c.iterations * operations_per_step;
    r.metadata["mode"] = c.latency ? "latency" : "throughput";
    r.metadata["clock_scope"] = "clock64 interval includes loop overhead and warp scheduling; not an isolated SASS instruction latency";
    r.metadata["validation_scope"] = "deterministic recurrence; at most 256 final values at stratified lanes, including every ILP chain and packed component";
    r.metrics["iterations"] = c.iterations;
    r.metrics["ilp"] = c.ilp;
    r.metrics["threads"] = c.threads;
    r.metrics["blocks"] = c.blocks;
    auto clocks = copy_output(clock, stream);
    std::sort(clocks.begin(), clocks.end());
    const auto best = clocks.front();
    r.metrics["minimum_clock_cycles_per_step"] = double(best) / c.iterations;
    r.metrics["minimum_thread_clock_cycles"] = double(best);
    r.metrics["median_thread_clock_cycles"] = clocks.size() % 2 ? double(clocks[clocks.size() / 2]) :
        (double(clocks[clocks.size() / 2 - 1]) + double(clocks[clocks.size() / 2])) / 2;
    r.metrics["maximum_thread_clock_cycles"] = double(clocks.back());
    r.metadata["clock_step_definition"] = "one loop iteration across all ILP chains, in the last measured launch; never divide by a nominal GPU clock to invent a latency";
}

void validate_number(result& r, double got, double expected, double atol, double rtol) {
    const double error = std::abs(got - expected);
    const double relative = error / std::max(std::abs(expected), 1e-30);
    r.max_abs_error = std::max(r.max_abs_error, std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
    r.max_rel_error = std::max(r.max_rel_error, std::isfinite(relative) ? relative : std::numeric_limits<double>::infinity());
    if (!std::isfinite(got) || !std::isfinite(expected) || error > atol + rtol * std::abs(expected)) {
        r.validation = "failed";
        r.status = "validation_failed";
        r.reason = "GPU recurrence did not match the independent host reference";
    }
}

template<class U>
void validate_integer_bits(result& r, U got, U expected, const char* reason) {
    // Subtract before conversion to double: adjacent u64/u128 bit patterns can
    // round to the same floating value, hiding an exact integer mismatch.
    if (got == expected) return;
    const U difference = got > expected ? got - expected : expected - got;
    r.max_abs_error = std::max(r.max_abs_error, static_cast<double>(difference));
    r.max_rel_error = std::max(r.max_rel_error, static_cast<double>(difference) /
                                             std::max(1.0, static_cast<double>(expected)));
    r.metrics["bit_pattern_mismatches"] += 1;
    r.metadata["error_metric"] = "unsigned encoded bit-pattern distance; exact equality determines integer acceptance";
    r.validation = "failed";
    r.status = "validation_failed";
    r.reason = reason;
}

template <typename U>
__host__ __device__ U integer_seed(size_t lane, int chain) {
    const uint64_t mixed = (uint64_t(lane) + 1) * UINT64_C(0x9e3779b97f4a7c15)
                         + uint64_t(chain + 1) * UINT64_C(0xd1b54a32d192ed03);
    return U(mixed);
}

template <typename T>
__host__ __device__ T signed_from_bits(typename std::make_unsigned<T>::type u) {
    using U = typename std::make_unsigned<T>::type;
    if constexpr (std::is_unsigned<T>::value) return u;
    const U positive_limit = U(std::numeric_limits<T>::max());
    if (u <= positive_limit) return T(u);
    return T(-1 - T(U(~u)));
}

template <typename T, int_op Op>
__device__ __forceinline__ typename std::make_unsigned<T>::type integer_device(
    typename std::make_unsigned<T>::type a,
    typename std::make_unsigned<T>::type b,
    typename std::make_unsigned<T>::type c) {
    using U = typename std::make_unsigned<T>::type;
    if constexpr (sizeof(T) <= 4) {
        uint32_t x = uint32_t(a), y = uint32_t(b), z = uint32_t(c), o = 0;
        int32_t sx = int32_t(signed_from_bits<T>(a)), sy = int32_t(signed_from_bits<T>(b));
        if constexpr (Op == int_op::add) asm volatile("add.u32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(y));
        if constexpr (Op == int_op::sub) asm volatile("sub.u32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(y));
        if constexpr (Op == int_op::mul) asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(y));
        if constexpr (Op == int_op::mad) asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(o) : "r"(x), "r"(y), "r"(z));
        if constexpr (Op == int_op::div) {
            if constexpr (std::is_signed<T>::value) asm volatile("div.s32 %0, %1, %2;" : "=r"(o) : "r"(sx), "r"(sy));
            else asm volatile("div.u32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(y));
        }
        if constexpr (Op == int_op::rem) {
            if constexpr (std::is_signed<T>::value) asm volatile("rem.s32 %0, %1, %2;" : "=r"(o) : "r"(sx), "r"(sy));
            else asm volatile("rem.u32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(y));
        }
        if constexpr (Op == int_op::min) {
            if constexpr (std::is_signed<T>::value) asm volatile("min.s32 %0, %1, %2;" : "=r"(o) : "r"(sx), "r"(sy));
            else asm volatile("min.u32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(y));
        }
        if constexpr (Op == int_op::max) {
            if constexpr (std::is_signed<T>::value) asm volatile("max.s32 %0, %1, %2;" : "=r"(o) : "r"(sx), "r"(sy));
            else asm volatile("max.u32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(y));
        }
        if constexpr (Op == int_op::bit_and) asm volatile("and.b32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(y));
        if constexpr (Op == int_op::bit_or) asm volatile("or.b32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(y));
        if constexpr (Op == int_op::bit_xor) asm volatile("xor.b32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(y));
        if constexpr (Op == int_op::shl) asm volatile("shl.b32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(uint32_t(y & (sizeof(T) * 8 - 1))));
        if constexpr (Op == int_op::shr) {
            if constexpr (std::is_signed<T>::value) asm volatile("shr.s32 %0, %1, %2;" : "=r"(o) : "r"(sx), "r"(uint32_t(y & (sizeof(T) * 8 - 1))));
            else asm volatile("shr.u32 %0, %1, %2;" : "=r"(o) : "r"(x), "r"(uint32_t(y & (sizeof(T) * 8 - 1))));
        }
        return U(o);
    } else {
        unsigned long long x = a, y = b, z = c, o = 0;
        long long sx = signed_from_bits<T>(a), sy = signed_from_bits<T>(b);
        if constexpr (Op == int_op::add) asm volatile("add.u64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
        if constexpr (Op == int_op::sub) asm volatile("sub.u64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
        if constexpr (Op == int_op::mul) asm volatile("mul.lo.u64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
        if constexpr (Op == int_op::mad) {
            asm volatile("mul.lo.u64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
            asm volatile("add.u64 %0, %1, %2;" : "=l"(o) : "l"(o), "l"(z));
        }
        if constexpr (Op == int_op::div) {
            if constexpr (std::is_signed<T>::value) asm volatile("div.s64 %0, %1, %2;" : "=l"(o) : "l"(sx), "l"(sy));
            else asm volatile("div.u64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
        }
        if constexpr (Op == int_op::rem) {
            if constexpr (std::is_signed<T>::value) asm volatile("rem.s64 %0, %1, %2;" : "=l"(o) : "l"(sx), "l"(sy));
            else asm volatile("rem.u64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
        }
        if constexpr (Op == int_op::min) {
            if constexpr (std::is_signed<T>::value) asm volatile("min.s64 %0, %1, %2;" : "=l"(o) : "l"(sx), "l"(sy));
            else asm volatile("min.u64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
        }
        if constexpr (Op == int_op::max) {
            if constexpr (std::is_signed<T>::value) asm volatile("max.s64 %0, %1, %2;" : "=l"(o) : "l"(sx), "l"(sy));
            else asm volatile("max.u64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
        }
        if constexpr (Op == int_op::bit_and) asm volatile("and.b64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
        if constexpr (Op == int_op::bit_or) asm volatile("or.b64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
        if constexpr (Op == int_op::bit_xor) asm volatile("xor.b64 %0, %1, %2;" : "=l"(o) : "l"(x), "l"(y));
        if constexpr (Op == int_op::shl) asm volatile("shl.b64 %0, %1, %2;" : "=l"(o) : "l"(x), "r"(uint32_t(y & 63)));
        if constexpr (Op == int_op::shr) {
            if constexpr (std::is_signed<T>::value) asm volatile("shr.s64 %0, %1, %2;" : "=l"(o) : "l"(sx), "r"(uint32_t(y & 63)));
            else asm volatile("shr.u64 %0, %1, %2;" : "=l"(o) : "l"(x), "r"(uint32_t(y & 63)));
        }
        return U(o);
    }
}

template <typename T, int_op Op>
typename std::make_unsigned<T>::type integer_reference(typename std::make_unsigned<T>::type a,
                                                        typename std::make_unsigned<T>::type b,
                                                        typename std::make_unsigned<T>::type c) {
    using U = typename std::make_unsigned<T>::type;
    const auto sa = signed_from_bits<T>(a), sb = signed_from_bits<T>(b);
    if constexpr (Op == int_op::add) return U(uint64_t(a) + uint64_t(b));
    if constexpr (Op == int_op::sub) return U(uint64_t(a) - uint64_t(b));
    if constexpr (Op == int_op::mul) return U(uint64_t(a) * uint64_t(b));
    if constexpr (Op == int_op::mad) return U(uint64_t(a) * uint64_t(b) + uint64_t(c));
    if constexpr (Op == int_op::div) return U(sa / sb);
    if constexpr (Op == int_op::rem) return U(sa % sb);
    if constexpr (Op == int_op::min) return sa < sb ? a : b;
    if constexpr (Op == int_op::max) return sa > sb ? a : b;
    if constexpr (Op == int_op::bit_and) return U(a & b);
    if constexpr (Op == int_op::bit_or) return U(a | b);
    if constexpr (Op == int_op::bit_xor) return U(a ^ b);
    if constexpr (Op == int_op::shl) return U(uint64_t(a) << (b & (sizeof(T) * 8 - 1)));
    if constexpr (Op == int_op::shr) {
        const unsigned shift = b & (sizeof(T) * 8 - 1);
        if constexpr (std::is_signed<T>::value) {
            // Explicit two's-complement arithmetic shift, independent of host signed >> rules.
            if (sa < 0 && shift) return U((uint64_t(a) >> shift) | (UINT64_MAX << (sizeof(T) * 8 - shift)));
        }
        return U(uint64_t(a) >> shift);
    }
    return U(0);
}

template <typename T, int_op Op, int Ilp>
__global__ void integer_kernel(typename std::make_unsigned<T>::type* output, unsigned long long* clocks,
                               int iterations, typename std::make_unsigned<T>::type b,
                               typename std::make_unsigned<T>::type c) {
    using U = typename std::make_unsigned<T>::type;
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    U values[Ilp];
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) values[j] = integer_seed<U>(lane, j);
    const auto begin = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int j = 0; j < Ilp; ++j) values[j] = integer_device<T, Op>(values[j], b, c);
    }
    const auto end = clock64();
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) output[lane * Ilp + j] = values[j];
    clocks[lane] = end - begin;
}

template <typename T, int_op Op, int Ilp>
void integer_case(context& ctx, result& r, const micro_config& config) {
    using U = typename std::make_unsigned<T>::type;
    const U b = U(ctx.opts.integer("operand", 13));
    const U c = U(ctx.opts.integer("addend", 17));
    if constexpr (Op == int_op::div || Op == int_op::rem) {
        if (signed_from_bits<T>(b) <= 0)
            throw std::invalid_argument("division/rem operand must be positive after conversion; avoids divide-by-zero and MIN/-1");
    }
    ctx.memory_guard(config.lanes() * (Ilp * sizeof(U) + sizeof(unsigned long long)));
    device_buffer<U> output(config.lanes() * Ilp);
    device_buffer<unsigned long long> clocks(config.lanes());
    auto launch = [&] { integer_kernel<T, Op, Ilp><<<config.blocks, config.threads, 0, ctx.stream>>>(output.get(), clocks.get(), config.iterations, b, c); };
    r.samples_ms = ctx.measure(launch);
    auto actual = copy_output(output, ctx.stream);
    r.validation = "passed";
    size_t checked = 0;
    for (size_t q : micro_sample_indices(config.lanes(), Ilp)) {
        U expected = integer_seed<U>(q / Ilp, int(q % Ilp));
        for (int i = 0; i < config.iterations; ++i) expected = integer_reference<T, Op>(expected, b, c);
        validate_integer_bits(r, actual[q], expected, "integer bit pattern differs from modulo-arithmetic host reference");
        ++checked;
    }
    r.execution = sizeof(T) < 4 ? "promoted_ptx_with_width_wrap" : "requested_ptx";
    r.unit = "Gop/s";
    r.metadata["work_domain"] = "integer";
    r.metadata["integer_semantics"] = "add/sub/mul/mad wrap modulo 2^width; signed divide truncates toward zero; signed right shift sign-extends";
    r.metadata["machine_code_evidence"] = "PTX requests recorded; inspect cuobjdump output to establish actual SASS lowering";
    r.metrics["validation_values"] = double(checked);
    record_micro(r, config, clocks, ctx.stream, Op == int_op::mad ? 2 : 1);
}

#define ARCHBENCH_ILP_DISPATCH(FUNCTION, ...) \
    do { switch (config.ilp) { \
    case 1: FUNCTION<__VA_ARGS__, 1>(ctx, r, config); break; \
    case 2: FUNCTION<__VA_ARGS__, 2>(ctx, r, config); break; \
    case 4: FUNCTION<__VA_ARGS__, 4>(ctx, r, config); break; \
    case 8: FUNCTION<__VA_ARGS__, 8>(ctx, r, config); break; \
    } } while (0)

template <typename T>
void dispatch_integer(context& ctx, result& r, const micro_config& config) {
    const auto op = ctx.opts.get("op", "add");
    #define INT_CASE(NAME, ENUM) if (op == NAME) { ARCHBENCH_ILP_DISPATCH(integer_case, T, int_op::ENUM); return; }
    INT_CASE("add", add) INT_CASE("sub", sub) INT_CASE("mul", mul) INT_CASE("mad", mad)
    INT_CASE("div", div) INT_CASE("rem", rem) INT_CASE("min", min) INT_CASE("max", max)
    INT_CASE("and", bit_and) INT_CASE("or", bit_or) INT_CASE("xor", bit_xor) INT_CASE("shl", shl) INT_CASE("shr", shr)
    #undef INT_CASE
    skip(r, "unimplemented integer operation: " + op, "not_implemented");
}

template <typename T> __host__ __device__ double as_double(T x) { return double(x); }
template <> __host__ __device__ double as_double(__half x) { return double(__half2float(x)); }
template <> __host__ __device__ double as_double(__nv_bfloat16 x) { return double(__bfloat162float(x)); }

template<class T>
void validate_all_finite(result& r, const std::vector<T>& values) {
    size_t nonfinite = 0;
    for (T value : values) {
        if (!std::isfinite(as_double(value))) {
            ++nonfinite;
            validate_number(r, as_double(value), 0, 0, 0);
        }
    }
    r.metrics["nonfinite_output_values"] = static_cast<double>(nonfinite);
    r.metadata["all_downloaded_outputs_finite"] = nonfinite ? "false" : "true";
}

template <typename T> __host__ __device__ T from_double(double x) { return T(x); }
template <> __host__ __device__ __half from_double(double x) { return __float2half_rn(float(x)); }
template <> __host__ __device__ __nv_bfloat16 from_double(double x) { return __float2bfloat16_rn(float(x)); }

template <typename T>
__host__ __device__ T float_seed(size_t lane, int chain) {
    return from_double<T>(0.5 + double((lane + size_t(chain) * 7) % 32) / 64.0);
}

template <typename T, float_op Op>
__device__ __forceinline__ T floating_device(T x, T b, T c) {
    if constexpr (std::is_same<T, float>::value) {
        float y = 0;
        if constexpr (Op == float_op::add) asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(b));
        if constexpr (Op == float_op::sub) asm volatile("sub.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(b));
        if constexpr (Op == float_op::mul) asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(b));
        if constexpr (Op == float_op::fma) asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(x), "f"(b), "f"(c));
        if constexpr (Op == float_op::div) asm volatile("div.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(b));
        if constexpr (Op == float_op::sqrt) asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == float_op::min) asm volatile("min.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(b));
        if constexpr (Op == float_op::max) asm volatile("max.f32 %0, %1, %2;" : "=f"(y) : "f"(x), "f"(b));
        if constexpr (Op == float_op::abs) asm volatile("abs.f32 %0, %1;" : "=f"(y) : "f"(x));
        return y;
    } else if constexpr (std::is_same<T, double>::value) {
        double y = 0;
        if constexpr (Op == float_op::add) asm volatile("add.rn.f64 %0, %1, %2;" : "=d"(y) : "d"(x), "d"(b));
        if constexpr (Op == float_op::sub) asm volatile("sub.rn.f64 %0, %1, %2;" : "=d"(y) : "d"(x), "d"(b));
        if constexpr (Op == float_op::mul) asm volatile("mul.rn.f64 %0, %1, %2;" : "=d"(y) : "d"(x), "d"(b));
        if constexpr (Op == float_op::fma) asm volatile("fma.rn.f64 %0, %1, %2, %3;" : "=d"(y) : "d"(x), "d"(b), "d"(c));
        if constexpr (Op == float_op::div) asm volatile("div.rn.f64 %0, %1, %2;" : "=d"(y) : "d"(x), "d"(b));
        if constexpr (Op == float_op::sqrt) asm volatile("sqrt.rn.f64 %0, %1;" : "=d"(y) : "d"(x));
        if constexpr (Op == float_op::min) asm volatile("min.f64 %0, %1, %2;" : "=d"(y) : "d"(x), "d"(b));
        if constexpr (Op == float_op::max) asm volatile("max.f64 %0, %1, %2;" : "=d"(y) : "d"(x), "d"(b));
        if constexpr (Op == float_op::abs) asm volatile("abs.f64 %0, %1;" : "=d"(y) : "d"(x));
        return y;
    } else {
        // CUDA headers select valid PTX / synthesized BF16 sequences per target.
        // asm barriers make the recurrence visible even for idempotent operations.
        if constexpr (std::is_same<T, __half>::value) {
            unsigned short bits = __half_as_ushort(x);
            asm volatile("mov.b16 %0, %1;" : "=h"(bits) : "h"(bits));
            x = __ushort_as_half(bits);
        } else {
            unsigned short bits = __bfloat16_as_ushort(x);
            asm volatile("mov.b16 %0, %1;" : "=h"(bits) : "h"(bits));
            x = __ushort_as_bfloat16(bits);
        }
        if constexpr (Op == float_op::add) return __hadd(x, b);
        if constexpr (Op == float_op::sub) return __hsub(x, b);
        if constexpr (Op == float_op::mul) return __hmul(x, b);
        if constexpr (Op == float_op::fma) return __hfma(x, b, c);
        if constexpr (Op == float_op::div) return from_double<T>(float(as_double(x)) / float(as_double(b)));
        if constexpr (Op == float_op::sqrt) return from_double<T>(sqrtf(float(as_double(x))));
        if constexpr (Op == float_op::min) return __hmin(x, b);
        if constexpr (Op == float_op::max) return __hmax(x, b);
        if constexpr (Op == float_op::abs) return __habs(x);
    }
    return x;
}

template <typename T, float_op Op>
T floating_reference(T x, T b, T c) {
    const double a = as_double(x), q = as_double(b), z = as_double(c);
    if constexpr (Op == float_op::add) return from_double<T>(a + q);
    if constexpr (Op == float_op::sub) return from_double<T>(a - q);
    if constexpr (Op == float_op::mul) return from_double<T>(a * q);
    if constexpr (Op == float_op::fma) return from_double<T>(std::fma(a, q, z));
    if constexpr (Op == float_op::div) return from_double<T>(a / q);
    if constexpr (Op == float_op::sqrt) return from_double<T>(std::sqrt(a));
    if constexpr (Op == float_op::min) return a < q ? x : b;
    if constexpr (Op == float_op::max) return a > q ? x : b;
    if constexpr (Op == float_op::abs) return from_double<T>(std::abs(a));
    return x;
}

template <typename T, float_op Op, int Ilp>
__global__ void floating_kernel(T* output, unsigned long long* clocks, int iterations, T b, T c) {
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    T values[Ilp];
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) values[j] = float_seed<T>(lane, j);
    const auto begin = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int j = 0; j < Ilp; ++j) values[j] = floating_device<T, Op>(values[j], b, c);
    }
    const auto end = clock64();
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) output[lane * Ilp + j] = values[j];
    clocks[lane] = end - begin;
}

template <typename T, float_op Op, int Ilp>
void floating_case(context& ctx, result& r, const micro_config& config) {
    double default_b = 0.015625;
    if constexpr (Op == float_op::mul) default_b = 0.99609375;
    if constexpr (Op == float_op::fma) default_b = 0.5;
    if constexpr (Op == float_op::div) default_b = 1.00390625;
    if constexpr (Op == float_op::min || Op == float_op::max) default_b = 0.75;
    const T b = from_double<T>(ctx.opts.real("operand", default_b));
    const T c = from_double<T>(ctx.opts.real("addend", 0.25));
    if (!std::isfinite(as_double(b)) || !std::isfinite(as_double(c)) || (Op == float_op::div && as_double(b) == 0))
        throw std::invalid_argument("floating operands must be finite and divisor nonzero");
    ctx.memory_guard(config.lanes() * (Ilp * sizeof(T) + sizeof(unsigned long long)));
    device_buffer<T> output(config.lanes() * Ilp);
    device_buffer<unsigned long long> clocks(config.lanes());
    auto launch = [&] { floating_kernel<T, Op, Ilp><<<config.blocks, config.threads, 0, ctx.stream>>>(output.get(), clocks.get(), config.iterations, b, c); };
    r.samples_ms = ctx.measure(launch);
    auto actual = copy_output(output, ctx.stream);
    r.validation = "passed";
    validate_all_finite(r, actual);
    for (size_t q : micro_sample_indices(config.lanes(), Ilp)) {
        T expected = float_seed<T>(q / Ilp, int(q % Ilp));
        for (int i = 0; i < config.iterations; ++i) expected = floating_reference<T, Op>(expected, b, c);
        const double tol = std::is_same<T, double>::value ? 2e-13 : (sizeof(T) == 2 ? 2e-3 : 2e-6);
        validate_number(r, as_double(actual[q]), as_double(expected), tol, tol);
    }
    r.execution = sizeof(T) == 2 ? "cuda_intrinsic_lowering" : "requested_ptx";
    r.unit = "Gflop/s";
    r.metadata["work_domain"] = "floating_point";
    if (sizeof(T) == 2 && (Op == float_op::div || Op == float_op::sqrt)) r.execution = "emulated_via_f32";
    if (std::is_same<T, __nv_bfloat16>::value && ctx.prop.major < 9 &&
        (Op == float_op::add || Op == float_op::sub || Op == float_op::mul))
        r.metadata["lowering_note"] = "BF16 add/sub/mul PTX requires sm90; CUDA headers synthesize older-target operations";
    r.metadata["floating_inputs"] = "bounded finite deterministic seeds; arithmetic edges are separate correctness cases";
    record_micro(r, config, clocks, ctx.stream, Op == float_op::fma ? 2 : 1);
}

template <typename T>
void dispatch_floating(context& ctx, result& r, const micro_config& config) {
    const auto op = ctx.opts.get("op", "fma");
    #define FLOAT_CASE(NAME, ENUM) if (op == NAME) { ARCHBENCH_ILP_DISPATCH(floating_case, T, float_op::ENUM); return; }
    FLOAT_CASE("add", add) FLOAT_CASE("sub", sub) FLOAT_CASE("mul", mul) FLOAT_CASE("fma", fma)
    FLOAT_CASE("div", div) FLOAT_CASE("sqrt", sqrt) FLOAT_CASE("min", min) FLOAT_CASE("max", max) FLOAT_CASE("abs", abs)
    #undef FLOAT_CASE
    skip(r, "unimplemented scalar operation: " + op, "not_implemented");
}

template <typename U, bit_op Op>
__device__ __forceinline__ U bit_device(U x, U other) {
    if constexpr (sizeof(U) == 4) {
        uint32_t y = 0;
        if constexpr (Op == bit_op::popc) asm volatile("popc.b32 %0, %1;" : "=r"(y) : "r"(x));
        if constexpr (Op == bit_op::clz) asm volatile("clz.b32 %0, %1;" : "=r"(y) : "r"(x));
        if constexpr (Op == bit_op::brev) asm volatile("brev.b32 %0, %1;" : "=r"(y) : "r"(x));
        if constexpr (Op == bit_op::prmt) asm volatile("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(y) : "r"(x), "r"(other));
        if constexpr (Op == bit_op::lop3) asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(y) : "r"(x), "r"(other), "r"(0x12345678u));
        if constexpr (Op == bit_op::shf) asm volatile("shf.r.wrap.b32 %0, %1, %2, 7;" : "=r"(y) : "r"(x), "r"(other));
        return U(y);
    } else {
        unsigned long long y = 0, a = x;
        uint32_t small = 0;
        if constexpr (Op == bit_op::popc) { asm volatile("popc.b64 %0, %1;" : "=r"(small) : "l"(a)); y = small; }
        if constexpr (Op == bit_op::clz) { asm volatile("clz.b64 %0, %1;" : "=r"(small) : "l"(a)); y = small; }
        if constexpr (Op == bit_op::brev) asm volatile("brev.b64 %0, %1;" : "=l"(y) : "l"(a));
        return U(y);
    }
}

template <typename U, bit_op Op>
U bit_reference(U x, U other) {
    constexpr int bits = sizeof(U) * 8;
    if constexpr (Op == bit_op::popc) { U count = 0; for (int b = 0; b < bits; ++b) count += (x >> b) & U(1); return count; }
    if constexpr (Op == bit_op::clz) {
        U count = 0;
        for (int b = bits - 1; b >= 0 && ((x >> b) & U(1)) == 0; --b) ++count;
        return count;
    }
    if constexpr (Op == bit_op::brev) { U y = 0; for (int b = 0; b < bits; ++b) y |= U(((x >> b) & U(1)) << (bits - 1 - b)); return y; }
    if constexpr (Op == bit_op::prmt) return U((x & 0x0000ffffu) | ((other & 0x0000ffffu) << 16));
    if constexpr (Op == bit_op::lop3) return U(x ^ other ^ U(0x12345678u));
    if constexpr (Op == bit_op::shf) return U((x >> 7) | (other << 25));
    return 0;
}

template <typename U, bit_op Op, int Ilp>
__global__ void bit_kernel(U* output, unsigned long long* clocks, int iterations, U other) {
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    U values[Ilp];
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) values[j] = integer_seed<U>(lane, j);
    auto begin = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int j = 0; j < Ilp; ++j) values[j] = bit_device<U, Op>(values[j], other);
    }
    auto end = clock64();
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) output[lane * Ilp + j] = values[j];
    clocks[lane] = end - begin;
}

template <typename U, bit_op Op, int Ilp>
void bit_case(context& ctx, result& r, const micro_config& config) {
    const U other = U(ctx.opts.integer("operand", 0x5a87c3e1u));
    ctx.memory_guard(config.lanes() * (Ilp * sizeof(U) + sizeof(unsigned long long)));
    device_buffer<U> output(config.lanes() * Ilp);
    device_buffer<unsigned long long> clocks(config.lanes());
    r.samples_ms = ctx.measure([&] { bit_kernel<U, Op, Ilp><<<config.blocks, config.threads, 0, ctx.stream>>>(output.get(), clocks.get(), config.iterations, other); });
    auto actual = copy_output(output, ctx.stream);
    r.validation = "passed";
    for (size_t q : micro_sample_indices(config.lanes(), Ilp)) {
        U expected = integer_seed<U>(q / Ilp, int(q % Ilp));
        for (int i = 0; i < config.iterations; ++i) expected = bit_reference<U, Op>(expected, other);
        validate_integer_bits(r, actual[q], expected, "bit-operation reference mismatch");
    }
    r.execution = "requested_ptx";
    record_micro(r, config, clocks, ctx.stream);
    r.unit = "Gop/s";
    r.metadata["work_domain"] = "bit";
}

template <typename U>
void dispatch_bit(context& ctx, result& r, const micro_config& config) {
    const auto op = ctx.opts.get("op", "popc");
    #define BIT_CASE(NAME, ENUM) if (op == NAME) { ARCHBENCH_ILP_DISPATCH(bit_case, U, bit_op::ENUM); return; }
    BIT_CASE("popc", popc) BIT_CASE("clz", clz) BIT_CASE("brev", brev)
    if constexpr (sizeof(U) == 4) { BIT_CASE("prmt", prmt) BIT_CASE("lop3", lop3) BIT_CASE("shf", shf) }
    #undef BIT_CASE
    skip(r, "bit operation has no implemented form for this width", "not_implemented");
}

template <sfu_op Op, bool Approximate>
__device__ __forceinline__ float sfu_device(float x) {
    float y = 0;
    if constexpr (Approximate) {
        if constexpr (Op == sfu_op::rcp) asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == sfu_op::rsqrt) asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == sfu_op::sin) asm volatile("sin.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == sfu_op::cos) asm volatile("cos.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == sfu_op::exp2) asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
        if constexpr (Op == sfu_op::log2) asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    } else {
        if constexpr (Op == sfu_op::rcp) y = __fdiv_rn(1.0f, x);
        if constexpr (Op == sfu_op::rsqrt) y = __fdiv_rn(1.0f, __fsqrt_rn(x));
        if constexpr (Op == sfu_op::sin) y = sinf(x);
        if constexpr (Op == sfu_op::cos) y = cosf(x);
        if constexpr (Op == sfu_op::exp2) y = exp2f(x);
        if constexpr (Op == sfu_op::log2) y = log2f(x);
    }
    return y;
}

template <sfu_op Op>
float sfu_reference(float x) {
    if constexpr (Op == sfu_op::rcp) return 1.0f / x;
    if constexpr (Op == sfu_op::rsqrt) return 1.0f / std::sqrt(x);
    if constexpr (Op == sfu_op::sin) return std::sin(x);
    if constexpr (Op == sfu_op::cos) return std::cos(x);
    if constexpr (Op == sfu_op::exp2) return std::exp2(x);
    if constexpr (Op == sfu_op::log2) return std::log2(x);
    return 0;
}

template <sfu_op Op, bool Approximate, int Ilp>
__global__ void sfu_kernel(float* output, unsigned long long* clocks, int iterations) {
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    float values[Ilp];
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) values[j] = float_seed<float>(lane, j);
    auto begin = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int j = 0; j < Ilp; ++j) values[j] = sfu_device<Op, Approximate>(__fmaf_rn(values[j], 0.125f, 1.5f));
    }
    auto end = clock64();
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) output[lane * Ilp + j] = values[j];
    clocks[lane] = end - begin;
}

template <sfu_op Op, bool Approximate, int Ilp>
void sfu_case(context& ctx, result& r, const micro_config& config) {
    ctx.memory_guard(config.lanes() * (Ilp * sizeof(float) + sizeof(unsigned long long)));
    device_buffer<float> output(config.lanes() * Ilp);
    device_buffer<unsigned long long> clocks(config.lanes());
    r.samples_ms = ctx.measure([&] { sfu_kernel<Op, Approximate, Ilp><<<config.blocks, config.threads, 0, ctx.stream>>>(output.get(), clocks.get(), config.iterations); });
    auto actual = copy_output(output, ctx.stream);
    r.validation = "passed";
    validate_all_finite(r, actual);
    for (size_t q : micro_sample_indices(config.lanes(), Ilp)) {
        float expected = float_seed<float>(q / Ilp, int(q % Ilp));
        for (int i = 0; i < config.iterations; ++i) expected = sfu_reference<Op>(std::fma(expected, 0.125f, 1.5f));
        validate_number(r, actual[q], expected, Approximate ? 3e-5 : 5e-6, Approximate ? 3e-5 : 5e-6);
    }
    r.execution = Approximate ? "requested_approximate_ptx" : "cuda_math_library";
    r.unit = "Gcall/s";
    r.metadata["work_domain"] = "special_function";
    r.metadata["sfu_domain"] = "each iteration applies fma(x,0.125,1.5) before the SFU operation to keep recurrence finite";
    r.metadata["work_accounting"] = "one useful SFU evaluation per step; the domain-maintenance FMA and loop overhead are timed but not counted";
    record_micro(r, config, clocks, ctx.stream);
}

template <bool Approximate>
void dispatch_sfu(context& ctx, result& r, const micro_config& config) {
    const auto op = ctx.opts.get("op", "exp2");
    #define SFU_CASE(NAME, ENUM) if (op == NAME) { ARCHBENCH_ILP_DISPATCH(sfu_case, sfu_op::ENUM, Approximate); return; }
    SFU_CASE("rcp", rcp) SFU_CASE("rsqrt", rsqrt) SFU_CASE("sin", sin) SFU_CASE("cos", cos) SFU_CASE("exp2", exp2) SFU_CASE("log2", log2)
    #undef SFU_CASE
    skip(r, "unimplemented special function", "not_implemented");
}

template <typename Half, float_op Op, int Ilp>
__global__ void packed_float_kernel(Half* output, unsigned long long* clocks, int iterations,
                                    Half b, Half c) {
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    using Pair = typename std::conditional<std::is_same<Half, __half>::value, __half2, __nv_bfloat162>::type;
    Pair values[Ilp];
    Pair pb, pc;
    if constexpr (std::is_same<Half, __half>::value) { pb = __halves2half2(b, b); pc = __halves2half2(c, c); }
    else { pb = __halves2bfloat162(b, b); pc = __halves2bfloat162(c, c); }
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) {
        const Half lo = float_seed<Half>(lane * 2, j), hi = float_seed<Half>(lane * 2 + 1, j);
        if constexpr (std::is_same<Half, __half>::value) values[j] = __halves2half2(lo, hi);
        else values[j] = __halves2bfloat162(lo, hi);
    }
    auto begin = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int j = 0; j < Ilp; ++j) {
            if constexpr (Op == float_op::add) values[j] = __hadd2(values[j], pb);
            if constexpr (Op == float_op::mul) values[j] = __hmul2(values[j], pb);
            if constexpr (Op == float_op::fma) values[j] = __hfma2(values[j], pb, pc);
        }
    }
    auto end = clock64();
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) {
        if constexpr (std::is_same<Half, __half>::value) {
            output[(lane * Ilp + j) * 2] = __low2half(values[j]);
            output[(lane * Ilp + j) * 2 + 1] = __high2half(values[j]);
        } else {
            output[(lane * Ilp + j) * 2] = __low2bfloat16(values[j]);
            output[(lane * Ilp + j) * 2 + 1] = __high2bfloat16(values[j]);
        }
    }
    clocks[lane] = end - begin;
}

template <typename Half, float_op Op, int Ilp>
void packed_float_case(context& ctx, result& r, const micro_config& config) {
    const Half b = from_double<Half>(Op == float_op::add ? 0.015625 : (Op == float_op::mul ? 0.99609375 : 0.5));
    const Half c = from_double<Half>(0.25);
    ctx.memory_guard(config.lanes() * (2 * Ilp * sizeof(Half) + sizeof(unsigned long long)));
    device_buffer<Half> output(config.lanes() * Ilp * 2);
    device_buffer<unsigned long long> clocks(config.lanes());
    r.samples_ms = ctx.measure([&] { packed_float_kernel<Half, Op, Ilp><<<config.blocks, config.threads, 0, ctx.stream>>>(output.get(), clocks.get(), config.iterations, b, c); });
    auto actual = copy_output(output, ctx.stream);
    r.validation = "passed";
    validate_all_finite(r, actual);
    for (size_t q : micro_sample_indices(config.lanes(), 2 * Ilp)) {
        const size_t pair = q / 2;
        Half expected = float_seed<Half>((pair / Ilp) * 2 + q % 2, int(pair % Ilp));
        for (int i = 0; i < config.iterations; ++i) expected = floating_reference<Half, Op>(expected, b, c);
        validate_number(r, as_double(actual[q]), as_double(expected), 0.002, 0.002);
    }
    r.execution = "packed_cuda_intrinsic_lowering";
    r.unit = "Gflop/s";
    r.metadata["work_domain"] = "floating_point";
    r.metadata["packed_lanes"] = "2 independent values per 32-bit register";
    if (std::is_same<Half, __nv_bfloat16>::value && ctx.prop.major < 9 && Op != float_op::fma)
        r.metadata["lowering_note"] = "BF16 add2/mul2 are synthesized on pre-sm90 targets";
    record_micro(r, config, clocks, ctx.stream, Op == float_op::fma ? 4 : 2);
}

template <typename Half>
void dispatch_packed_float(context& ctx, result& r, const micro_config& config) {
    const auto op = ctx.opts.get("op", "fma");
    if (op == "add") { ARCHBENCH_ILP_DISPATCH(packed_float_case, Half, float_op::add); return; }
    if (op == "mul") { ARCHBENCH_ILP_DISPATCH(packed_float_case, Half, float_op::mul); return; }
    if (op == "fma") { ARCHBENCH_ILP_DISPATCH(packed_float_case, Half, float_op::fma); return; }
    skip(r, "unimplemented packed floating operation", "not_implemented");
}

template <bool SignedA, bool SignedB, bool Two, bool High>
__device__ __forceinline__ uint32_t dot_device(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t out = 0;
    #define DOT_INSTRUCTION(PREFIX, A, B) asm volatile(PREFIX "." A "." B " %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c))
    #define DOT_SIGN_DISPATCH(PREFIX) \
        if constexpr (SignedA && SignedB) { DOT_INSTRUCTION(PREFIX, "s32", "s32"); } \
        else if constexpr (SignedA && !SignedB) { DOT_INSTRUCTION(PREFIX, "s32", "u32"); } \
        else if constexpr (!SignedA && SignedB) { DOT_INSTRUCTION(PREFIX, "u32", "s32"); } \
        else { DOT_INSTRUCTION(PREFIX, "u32", "u32"); }
    if constexpr (!Two) { DOT_SIGN_DISPATCH("dp4a") }
    else if constexpr (!High) { DOT_SIGN_DISPATCH("dp2a.lo") }
    else { DOT_SIGN_DISPATCH("dp2a.hi") }
    #undef DOT_SIGN_DISPATCH
    #undef DOT_INSTRUCTION
    return out;
}

int32_t extract_signed(uint32_t data, int shift, int bits, bool is_signed) {
    const uint32_t mask = (uint32_t(1) << bits) - 1;
    const uint32_t v = (data >> shift) & mask;
    return is_signed && (v & (uint32_t(1) << (bits - 1))) ? int32_t(v) - int32_t(uint32_t(1) << bits) : int32_t(v);
}

template <bool SignedA, bool SignedB, bool Two, bool High, int Ilp>
__global__ void dot_kernel(uint32_t* output, unsigned long long* clocks, int iterations, uint32_t a, uint32_t b) {
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t values[Ilp];
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) values[j] = uint32_t((lane + j) % 31);
    auto begin = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int j = 0; j < Ilp; ++j) values[j] = dot_device<SignedA, SignedB, Two, High>(a, b, values[j]);
    }
    auto end = clock64();
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) output[lane * Ilp + j] = values[j];
    clocks[lane] = end - begin;
}

template <bool SignedA, bool SignedB, bool Two, bool High, int Ilp>
void dot_case(context& ctx, result& r, const micro_config& config) {
    const uint32_t a = Two ? 0xfffd0007u : 0x7ffd0301u;
    const uint32_t b = 0x0302ff04u;
    int64_t product_sum = 0;
    for (int i = 0; i < (Two ? 2 : 4); ++i) {
        const int aa = extract_signed(a, i * (Two ? 16 : 8), Two ? 16 : 8, SignedA);
        const int bb = extract_signed(b, (i + (Two && High ? 2 : 0)) * 8, 8, SignedB);
        product_sum += int64_t(aa) * bb;
    }
    ctx.memory_guard(config.lanes() * (Ilp * sizeof(uint32_t) + sizeof(unsigned long long)));
    device_buffer<uint32_t> output(config.lanes() * Ilp);
    device_buffer<unsigned long long> clocks(config.lanes());
    r.samples_ms = ctx.measure([&] { dot_kernel<SignedA, SignedB, Two, High, Ilp><<<config.blocks, config.threads, 0, ctx.stream>>>(output.get(), clocks.get(), config.iterations, a, b); });
    auto actual = copy_output(output, ctx.stream);
    r.validation = "passed";
    for (size_t q : micro_sample_indices(config.lanes(), Ilp)) {
        const uint32_t expected = uint32_t((q / Ilp + q % Ilp) % 31) + uint32_t(product_sum * config.iterations);
        validate_integer_bits(r, actual[q], expected, "packed integer dot-product mismatch");
    }
    r.execution = "requested_ptx";
    r.metadata["accumulator_semantics"] = "32-bit modulo arithmetic, including signed mixed-input paths";
    record_micro(r, config, clocks, ctx.stream, Two ? 4 : 8);
    r.unit = "Gop/s";
    r.metadata["work_domain"] = "integer";
}

template <bool SignedA, bool SignedB>
void dispatch_dot_kind(context& ctx, result& r, const micro_config& config) {
    const auto op = ctx.opts.get("op");
    if (op == "dp4a") { ARCHBENCH_ILP_DISPATCH(dot_case, SignedA, SignedB, false, false); return; }
    if (op == "dp2a_lo") { ARCHBENCH_ILP_DISPATCH(dot_case, SignedA, SignedB, true, false); return; }
    if (op == "dp2a_hi") { ARCHBENCH_ILP_DISPATCH(dot_case, SignedA, SignedB, true, true); return; }
    skip(r, "unimplemented packed integer operation", "not_implemented");
}

enum class rounding { rn, rz, rd, ru };

rounding parse_rounding(const options& opts) {
    const auto name = opts.get("round", "rn");
    if (name == "rn") return rounding::rn;
    if (name == "rz") return rounding::rz;
    if (name == "rd") return rounding::rd;
    if (name == "ru") return rounding::ru;
    throw std::invalid_argument("round must be rn, rz, rd, or ru");
}

template <typename Source, typename Destination>
__device__ __forceinline__ Destination convert_device(Source x, rounding round) {
    #define ROUNDED_CONVERSION(PREFIX, VALUE) \
        do { switch (round) { \
        case rounding::rn: return Destination(PREFIX##_rn(VALUE)); \
        case rounding::rz: return Destination(PREFIX##_rz(VALUE)); \
        case rounding::rd: return Destination(PREFIX##_rd(VALUE)); \
        case rounding::ru: return Destination(PREFIX##_ru(VALUE)); \
        } } while (0)
    if constexpr (std::is_integral<Source>::value && std::is_integral<Destination>::value) {
        return Destination(x);
    } else if constexpr (std::is_integral<Destination>::value) {
        if constexpr (std::is_same<Source, double>::value) {
            if constexpr (sizeof(Destination) <= 4 && std::is_signed<Destination>::value) { ROUNDED_CONVERSION(__double2int, x); }
            if constexpr (sizeof(Destination) <= 4 && !std::is_signed<Destination>::value) { ROUNDED_CONVERSION(__double2uint, x); }
            if constexpr (sizeof(Destination) == 8 && std::is_signed<Destination>::value) { ROUNDED_CONVERSION(__double2ll, x); }
            if constexpr (sizeof(Destination) == 8 && !std::is_signed<Destination>::value) { ROUNDED_CONVERSION(__double2ull, x); }
        } else {
            const float value = float(as_double(x));
            if constexpr (sizeof(Destination) <= 4 && std::is_signed<Destination>::value) { ROUNDED_CONVERSION(__float2int, value); }
            if constexpr (sizeof(Destination) <= 4 && !std::is_signed<Destination>::value) { ROUNDED_CONVERSION(__float2uint, value); }
            if constexpr (sizeof(Destination) == 8 && std::is_signed<Destination>::value) { ROUNDED_CONVERSION(__float2ll, value); }
            if constexpr (sizeof(Destination) == 8 && !std::is_signed<Destination>::value) { ROUNDED_CONVERSION(__float2ull, value); }
        }
    } else if constexpr (std::is_same<Destination, __half>::value) {
        const float value = float(as_double(x));
        ROUNDED_CONVERSION(__float2half, value);
    } else if constexpr (std::is_same<Destination, __nv_bfloat16>::value) {
        const float value = float(as_double(x));
        ROUNDED_CONVERSION(__float2bfloat16, value);
    } else if constexpr (std::is_same<Destination, float>::value) {
        if constexpr (std::is_integral<Source>::value) {
            if constexpr (sizeof(Source) <= 4 && std::is_signed<Source>::value) { ROUNDED_CONVERSION(__int2float, int(x)); }
            if constexpr (sizeof(Source) <= 4 && !std::is_signed<Source>::value) { ROUNDED_CONVERSION(__uint2float, unsigned(x)); }
            if constexpr (sizeof(Source) == 8 && std::is_signed<Source>::value) { ROUNDED_CONVERSION(__ll2float, (long long)x); }
            if constexpr (sizeof(Source) == 8 && !std::is_signed<Source>::value) { ROUNDED_CONVERSION(__ull2float, (unsigned long long)x); }
        } else if constexpr (std::is_same<Source, double>::value) { ROUNDED_CONVERSION(__double2float, x); }
        else return float(as_double(x));
    } else if constexpr (std::is_same<Destination, double>::value) {
        if constexpr (std::is_integral<Source>::value && sizeof(Source) == 8) {
            if constexpr (std::is_signed<Source>::value) { ROUNDED_CONVERSION(__ll2double, (long long)x); }
            else { ROUNDED_CONVERSION(__ull2double, (unsigned long long)x); }
        } else return as_double(x);
    }
    #undef ROUNDED_CONVERSION
    return from_double<Destination>(0);
}

template <typename Source, typename Destination>
__global__ void conversion_kernel(const Source* input, Destination* output, size_t size, rounding round) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size; i += size_t(gridDim.x) * blockDim.x)
        output[i] = convert_device<Source, Destination>(input[i], round);
}

template <typename Destination>
Destination conversion_reference(double x, rounding round) {
    if constexpr (std::is_integral<Destination>::value) {
        if (round == rounding::rn) x = std::nearbyint(x);
        else if (round == rounding::rz) x = std::trunc(x);
        else if (round == rounding::rd) x = std::floor(x);
        else x = std::ceil(x);
        return Destination(x);
    } else if constexpr (std::is_same<Destination, __half>::value) {
        const float f = float(x);
        if (round == rounding::rn) return __float2half_rn(f);
        if (round == rounding::rz) return __float2half_rz(f);
        if (round == rounding::rd) return __float2half_rd(f);
        return __float2half_ru(f);
    } else if constexpr (std::is_same<Destination, __nv_bfloat16>::value) {
        const float f = float(x);
        if (round == rounding::rn) return __float2bfloat16_rn(f);
        if (round == rounding::rz) return __float2bfloat16_rz(f);
        if (round == rounding::rd) return __float2bfloat16_rd(f);
        return __float2bfloat16_ru(f);
    } else if constexpr (std::is_same<Destination, float>::value) {
        float value = float(x);
        if (round == rounding::rd && double(value) > x) value = std::nextafter(value, -std::numeric_limits<float>::infinity());
        if (round == rounding::ru && double(value) < x) value = std::nextafter(value, std::numeric_limits<float>::infinity());
        if (round == rounding::rz && ((x > 0 && double(value) > x) || (x < 0 && double(value) < x))) value = std::nextafter(value, 0.0f);
        return value;
    } else return Destination(x);
}

template <typename Source, typename Destination>
void conversion_case(context& ctx, result& r) {
    const int64_t signed_size = ctx.opts.integer("size", 1 << 20);
    if (signed_size <= 0 || signed_size > (int64_t(1) << 34)) throw std::invalid_argument("conversion size must be 1..2^34 elements");
    const size_t size = size_t(signed_size);
    const rounding round = parse_rounding(ctx.opts);
    ctx.memory_guard(size * (sizeof(Source) + sizeof(Destination)));
    device_buffer<Source> input(size);
    device_buffer<Destination> output(size);
    std::vector<Source> host_input(size);
    for (size_t i = 0; i < size; ++i) {
        double value = double(int(i % 257) - 128) * 0.137 + double(i % 7) * 0.0001;
        if constexpr (std::is_unsigned<Source>::value || std::is_unsigned<Destination>::value) value = std::abs(value);
        host_input[i] = from_double<Source>(value);
    }
    CUDA_CHECK(cudaMemcpyAsync(input.get(), host_input.data(), size * sizeof(Source), cudaMemcpyHostToDevice, ctx.stream));
    const int blocks = int(std::min<size_t>((size + 255) / 256, 65535));
    r.samples_ms = ctx.measure([&] { conversion_kernel<Source, Destination><<<blocks, 256, 0, ctx.stream>>>(input.get(), output.get(), size, round); });
    auto actual = copy_output(output, ctx.stream);
    r.validation = "passed";
    for (size_t i = 0; i < size; ++i) {
        Destination expected;
        if constexpr (std::is_integral<Source>::value && std::is_integral<Destination>::value) expected = Destination(host_input[i]);
        else expected = conversion_reference<Destination>(as_double(host_input[i]), round);
        if (as_double(actual[i]) != as_double(expected)) {
            r.validation = "failed"; r.status = "validation_failed";
            r.reason = "typed conversion differs from host reference";
            r.max_abs_error = std::max(r.max_abs_error, std::abs(as_double(actual[i]) - as_double(expected)));
        }
    }
    r.work_per_iteration = double(size);
    r.bytes_per_iteration = double(size) * (sizeof(Source) + sizeof(Destination));
    r.unit = "Gconversion/s";
    r.metadata["work_domain"] = "conversion";
    r.execution = std::is_same<Source, Destination>::value ? "identity_copy" : "cuda_intrinsic_conversion";
    r.metadata["validation_scope"] = "every element; finite values inside destination integer range; no out-of-range C++ casts";
    r.metadata["timing_scope"] = "typed input read, conversion, typed output store; memory traffic included";
    if constexpr (sizeof(Destination) == 2 && !std::is_integral<Destination>::value)
        r.metadata["conversion_route"] = "source -> f32 -> low-precision destination; f64 uses a documented f32 intermediate";
    if constexpr (std::is_integral<Source>::value && std::is_integral<Destination>::value)
        r.metadata["rounding"] = "irrelevant for integral-to-integral conversion";
}

template <typename Source>
void dispatch_conversion_destination(context& ctx, result& r) {
    const auto dtype = ctx.opts.get("dtype-b", "f32");
    #define CONVERT_DEST(NAME, TYPE) if (dtype == NAME) { conversion_case<Source, TYPE>(ctx, r); return; }
    CONVERT_DEST("i8", int8_t) CONVERT_DEST("u8", uint8_t) CONVERT_DEST("i16", int16_t) CONVERT_DEST("u16", uint16_t)
    CONVERT_DEST("i32", int32_t) CONVERT_DEST("u32", uint32_t) CONVERT_DEST("i64", int64_t) CONVERT_DEST("u64", uint64_t)
    CONVERT_DEST("f16", __half) CONVERT_DEST("bf16", __nv_bfloat16) CONVERT_DEST("f32", float) CONVERT_DEST("f64", double)
    #undef CONVERT_DEST
    skip(r, "conversion destination is not implemented", "not_implemented");
}

enum class tensor_format { f16, bf16, tf32, f64, i8, u8, i4, u4, b1_and, b1_xor, e2m1, e2m3, e3m2, e4m3, e5m2 };

constexpr bool narrow_format(tensor_format t) { return t >= tensor_format::e2m1; }
constexpr bool fp8_format(tensor_format t) { return t == tensor_format::e4m3 || t == tensor_format::e5m2; }
constexpr bool integer_format(tensor_format t) { return t >= tensor_format::i8 && t <= tensor_format::b1_xor; }
constexpr bool binary_format(tensor_format t) { return t == tensor_format::b1_and || t == tensor_format::b1_xor; }

#include "archbench/mma_narrow.cuh"

template <tensor_format A, tensor_format B>
struct fp8_mma;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#define ARCHBENCH_FP8_PAIR(A, ATYPE, B, BTYPE) \
template <> struct fp8_mma<tensor_format::A, tensor_format::B> { \
    __device__ __forceinline__ static void step(float* d, uint32_t a, uint32_t b) { \
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32." ATYPE "." BTYPE ".f32 " \
            "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};" \
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(b)); \
    } \
    __device__ __forceinline__ static void step_half(uint32_t* d, uint32_t a, uint32_t b) { \
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f16." ATYPE "." BTYPE ".f16 " \
            "{%0,%1}, {%2,%2,%2,%2}, {%3,%3}, {%0,%1};" \
            : "+r"(d[0]), "+r"(d[1]) : "r"(a), "r"(b)); \
    } \
};
#else
#define ARCHBENCH_FP8_PAIR(A, ATYPE, B, BTYPE) \
template <> struct fp8_mma<tensor_format::A, tensor_format::B> { \
    __device__ __forceinline__ static void step(float*, uint32_t, uint32_t) {} \
    __device__ __forceinline__ static void step_half(uint32_t*, uint32_t, uint32_t) {} \
};
#endif
ARCHBENCH_FP8_PAIR(e4m3, "e4m3", e4m3, "e4m3")
ARCHBENCH_FP8_PAIR(e4m3, "e4m3", e5m2, "e5m2")
ARCHBENCH_FP8_PAIR(e5m2, "e5m2", e4m3, "e4m3")
ARCHBENCH_FP8_PAIR(e5m2, "e5m2", e5m2, "e5m2")
#undef ARCHBENCH_FP8_PAIR

template <tensor_format A, tensor_format B, int Scale>
__device__ __forceinline__ void native_float_mma(float* d, uint32_t a, uint32_t b, uint32_t sa, uint32_t sb) {
    if constexpr (A == tensor_format::f16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(b));
    } else if constexpr (A == tensor_format::bf16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(b));
    } else if constexpr (A == tensor_format::tf32) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(b));
    } else if constexpr (Scale == 2 || Scale == 3) {
        packed_fp4_mma<Scale == 3>(d, a, b, sa, sb);
    } else if constexpr (Scale == 1) {
        narrow_mma<A, B, true>::step(d, a, b, sa, sb);
    } else if constexpr (fp8_format(A) && fp8_format(B)) {
        fp8_mma<A, B>::step(d, a, b);
    } else {
        narrow_mma<A, B, false>::step(d, a, b, sa, sb);
    }
}

template <tensor_format A, tensor_format B>
__device__ __forceinline__ void native_integer_mma(uint32_t* d, uint32_t a, uint32_t b) {
    #define INT_MMA(SHAPE, ATYPE, BTYPE) asm volatile("mma.sync.aligned." SHAPE ".row.col.s32." ATYPE "." BTYPE ".s32 " \
        "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};" \
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a), "r"(b))
    if constexpr (A == tensor_format::i8 && B == tensor_format::i8) { INT_MMA("m16n8k32", "s8", "s8"); }
    if constexpr (A == tensor_format::i8 && B == tensor_format::u8) { INT_MMA("m16n8k32", "s8", "u8"); }
    if constexpr (A == tensor_format::u8 && B == tensor_format::i8) { INT_MMA("m16n8k32", "u8", "s8"); }
    if constexpr (A == tensor_format::u8 && B == tensor_format::u8) { INT_MMA("m16n8k32", "u8", "u8"); }
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1200 || defined(ARCHBENCH_SM120_INT4_MMA))
    if constexpr (A == tensor_format::i4 && B == tensor_format::i4) { INT_MMA("m16n8k64", "s4", "s4"); }
    if constexpr (A == tensor_format::i4 && B == tensor_format::u4) { INT_MMA("m16n8k64", "s4", "u4"); }
    if constexpr (A == tensor_format::u4 && B == tensor_format::i4) { INT_MMA("m16n8k64", "u4", "s4"); }
    if constexpr (A == tensor_format::u4 && B == tensor_format::u4) { INT_MMA("m16n8k64", "u4", "u4"); }
    #endif
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1200 || defined(ARCHBENCH_SM120_BINARY_MMA))
    if constexpr (A == tensor_format::b1_and) {
        asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
            "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a), "r"(b));
    }
    if constexpr (A == tensor_format::b1_xor) {
        asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc "
            "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};"
            : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a), "r"(b));
    }
    #endif
    #undef INT_MMA
}

__device__ __forceinline__ void native_double_mma(double* d, double a, double b) {
#if defined(__CUDA_ARCH__) && defined(ARCHBENCH_HAS_FP64_MMA)
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};" : "+d"(d[0]), "+d"(d[1]) : "d"(a), "d"(b));
#endif
}

constexpr int tensor_k(tensor_format format, int scale) {
    if (scale == 2 || scale == 3) return 64;
    if (format == tensor_format::f16 || format == tensor_format::bf16) return 16;
    if (format == tensor_format::tf32) return 8;
    if (format == tensor_format::f64) return 4;
    if (format == tensor_format::i4 || format == tensor_format::u4) return 64;
    if (binary_format(format)) return 256;
    return 32;
}

uint32_t packed_positive_one(tensor_format format, int scale) {
    if ((scale == 2 || scale == 3) && format == tensor_format::e2m1) return 0x22222222u;
    switch (format) {
    case tensor_format::f16: return 0x3c003c00u;
    case tensor_format::bf16: return 0x3f803f80u;
    case tensor_format::tf32: return 0x3f800000u;
    case tensor_format::i8: case tensor_format::u8: return 0x01010101u;
    case tensor_format::i4: case tensor_format::u4: return 0x11111111u;
    case tensor_format::b1_and: case tensor_format::b1_xor: return 0xffffffffu;
    case tensor_format::e2m1: return 0x08080808u;
    case tensor_format::e2m3: return 0x08080808u;
    case tensor_format::e3m2: return 0x0c0c0c0cu;
    case tensor_format::e4m3: return 0x38383838u;
    case tensor_format::e5m2: return 0x3c3c3c3cu;
    default: return 0;
    }
}

__host__ __device__ uint32_t negate_packed_one(uint32_t packed, tensor_format format, int scale) {
    if ((scale == 2 || scale == 3) && format == tensor_format::e2m1) return packed | 0x88888888u;
    switch (format) {
    case tensor_format::f16: case tensor_format::bf16: return packed | 0x80008000u;
    case tensor_format::tf32: return packed | 0x80000000u;
    case tensor_format::i8: case tensor_format::i4: return 0xffffffffu;
    case tensor_format::e2m1: case tensor_format::e2m3: case tensor_format::e3m2: return packed | 0x20202020u;
    case tensor_format::e4m3: case tensor_format::e5m2: return packed | 0x80808080u;
    default: return packed;
    }
}

void record_tensor_work(result& r, const micro_config& config, int m, int n, int k,
                        int reset_every, bool binary, bool emulated) {
    const auto counts = tensor_model::count_work(config.lanes(), config.ilp, config.iterations, m, n, k, binary);
    r.work_per_iteration = static_cast<double>(counts.operations);
    r.metadata["exact_work_per_launch"] = std::to_string(counts.operations);
    r.metadata["exact_logical_mma_tiles_per_launch"] = std::to_string(counts.logical_tiles);
    r.metadata["work_count_definition"] = binary ?
        "warps * independent_chains * iterations * M*N*K bit pairs" :
        "warps * independent_chains * iterations * 2*M*N*K arithmetic operations; one MAC counts as two";
    r.metadata["instruction_count_evidence"] = emulated ?
        "logical MMA-equivalent tiles only; this backend issues scalar arithmetic, not MMA instructions" :
        "requested warp-level PTX MMA count, not a measured SASS instruction counter";
    r.metadata["accumulation_policy"] = reset_every ? "periodic_reset" : "growing";
    r.metadata["reset_work_accounting"] =
        "all MMA/scalar-dot iterations count; accumulator clears and segment-loop overhead are timed but excluded from useful arithmetic work";
    r.metrics["logical_mma_tiles_per_launch"] = static_cast<double>(counts.logical_tiles);
    r.metrics["operations_per_mma_tile"] = static_cast<double>(counts.operations_per_tile);
    r.metrics["warps_per_launch"] = static_cast<double>(counts.warps);
    r.metrics["reset_every"] = reset_every;
    r.metrics["accumulator_resets_per_chain"] = static_cast<double>(tensor_model::reset_events(config.iterations, reset_every));
    r.metrics["final_segment_iterations"] = tensor_model::final_segment_iterations(config.iterations, reset_every);
    r.metrics["longest_segment_iterations"] = tensor_model::largest_segment_iterations(config.iterations, reset_every);
}

template<tensor_format A, bool Emulated>
void record_tensor_codegen(result& r) {
    std::string evidence = Emulated ? "not_applicable" : "not_inspected";
    if constexpr (!Emulated) {
        if constexpr (A == tensor_format::f64) {
            #ifdef ARCHBENCH_FP64_MMA_CODEGEN
            evidence = ARCHBENCH_FP64_MMA_CODEGEN;
            #endif
        } else if constexpr (A == tensor_format::i4 || A == tensor_format::u4) {
            #ifdef ARCHBENCH_INT4_MMA_CODEGEN
            evidence = ARCHBENCH_INT4_MMA_CODEGEN;
            #endif
        } else if constexpr (binary_format(A)) {
            #ifdef ARCHBENCH_BINARY_MMA_CODEGEN
            evidence = ARCHBENCH_BINARY_MMA_CODEGEN;
            #endif
        }
        if (evidence == "compiler_expanded_mma" || evidence == "scalar_expansion")
            r.execution = "compiler_expanded_tensor_ptx";
    }
    r.metadata["tensor_codegen_evidence"] = evidence;
    r.metadata["tensor_codegen_evidence_scope"] = Emulated ?
        "scalar software baseline; no native MMA request" :
        "standalone exact-instruction compiler probe SASS, when available; neither runtime support nor the complete benchmark kernel instruction count is established by this probe";
}

template <tensor_format A, tensor_format B, int Scale, bool Emulated, int Ilp>
__global__ void tensor_kernel(
    typename std::conditional<A == tensor_format::f64, double,
        typename std::conditional<integer_format(A), uint32_t, float>::type>::type* output,
    unsigned long long* clocks, int iterations, int reset_every, uint32_t packed_a, uint32_t packed_b,
    uint32_t scale_a, uint32_t scale_b, float scale_product, bool alternate_sign) {
    using Accumulator = typename std::conditional<A == tensor_format::f64, double,
        typename std::conditional<integer_format(A), uint32_t, float>::type>::type;
    constexpr int count = A == tensor_format::f64 ? 2 : 4;
    constexpr int k = tensor_k(A, Scale);
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool negative = alternate_sign && ((lane / 32) & 1);
    const uint32_t a = negative ? negate_packed_one(packed_a, A, Scale) : packed_a;
    const uint32_t b = packed_b;
    Accumulator values[Ilp][count] = {};
    const auto begin = clock64();
    for (int completed = 0; completed < iterations;) {
        if (completed) {
            #pragma unroll
            for (int j = 0; j < Ilp; ++j) {
                #pragma unroll
                for (int v = 0; v < count; ++v) values[j][v] = 0;
            }
        }
        const int remaining = iterations - completed;
        const int segment = reset_every && reset_every < remaining ? reset_every : remaining;
        #pragma unroll 1
        for (int it = 0; it < segment; ++it) {
            #pragma unroll
            for (int j = 0; j < Ilp; ++j) {
                if constexpr (!Emulated) {
                    if constexpr (A == tensor_format::f64) native_double_mma(values[j], negative ? -1.0 : 1.0, 1.0);
                    else if constexpr (integer_format(A)) native_integer_mma<A, B>(values[j], a, b);
                    else native_float_mma<A, B, Scale>(values[j], a, b, scale_a, scale_b);
                } else {
                    #pragma unroll 1
                    for (int q = 0; q < k; ++q) {
                        #pragma unroll
                        for (int v = 0; v < count; ++v) {
                            if constexpr (integer_format(A)) {
                                values[j][v] = integer_device<int32_t, int_op::mad>(negative ? 0xffffffffu : 1u, 1u, values[j][v]);
                            } else {
                                values[j][v] = floating_device<Accumulator, float_op::fma>(
                                    negative ? Accumulator(-scale_product) : Accumulator(scale_product), Accumulator(1), values[j][v]);
                            }
                        }
                    }
                }
            }
        }
        completed += segment;
    }
    const auto end = clock64();
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) {
        #pragma unroll
        for (int v = 0; v < count; ++v) output[(lane * Ilp + j) * count + v] = values[j][v];
    }
    clocks[lane] = end - begin;
}

template <tensor_format A, tensor_format B, int Scale, bool Emulated, int Ilp>
void tensor_case(context& ctx, result& r, const micro_config& config) {
    constexpr int count = A == tensor_format::f64 ? 2 : 4;
    constexpr int m = A == tensor_format::f64 ? 8 : 16;
    constexpr int n = 8;
    constexpr int k = tensor_k(A, Scale);
    using Accumulator = typename std::conditional<A == tensor_format::f64, double,
        typename std::conditional<integer_format(A), uint32_t, float>::type>::type;
    const int reset_every = tensor_model::reset_interval(ctx.opts.integer("reset-every", 0));
    const int final_iterations = tensor_model::final_segment_iterations(config.iterations, reset_every);
    if constexpr (Emulated && std::is_same<Accumulator, float>::value) {
        if (uint64_t(k) * uint64_t(tensor_model::largest_segment_iterations(config.iterations, reset_every)) > (uint64_t(1) << 24))
            throw std::invalid_argument("emulated f32 tensor segment exceeds 2^24 scalar additions; use reset-every or lower iterations");
    }
    const auto pattern = ctx.opts.get("pattern", "ones");
    if (pattern != "ones" && pattern != "alternating_sign") throw std::invalid_argument("tensor pattern must be ones or alternating_sign");
    const bool alternate = pattern == "alternating_sign";
    if (alternate && (A == tensor_format::u8 || A == tensor_format::u4 || binary_format(A))) {
        skip(r, "alternating_sign needs a signed floating/integer A operand"); return;
    }
    if constexpr (!Emulated) {
        if constexpr (A == tensor_format::f64) {
            #ifndef ARCHBENCH_HAS_FP64_MMA
            skip(r, "FP64 MMA was not enabled by the target-specific compiler probe; the emulated backend remains available"); return;
            #endif
        }
        if constexpr (A == tensor_format::i4 || A == tensor_format::u4) {
            #ifndef ARCHBENCH_SM120_INT4_MMA
            if (ctx.prop.major >= 12) { skip(r, "INT4 MMA was not enabled by this SM120 target's compiler probe; the emulated backend remains available"); return; }
            #endif
        }
        if constexpr (binary_format(A)) {
            #ifndef ARCHBENCH_SM120_BINARY_MMA
            if (ctx.prop.major >= 12) { skip(r, "binary MMA was not enabled by this SM120 target's compiler probe; the emulated backend remains available"); return; }
            #endif
        }
        if constexpr ((narrow_format(A) && !(fp8_format(A) && fp8_format(B))) || Scale != 0) {
            bool suffix_a = false;
            #ifdef ARCHBENCH_ARCH_SUFFIX_A
            suffix_a = true;
            #endif
            if (ctx.prop.major != 12 || ctx.prop.minor != 0 || !suffix_a) {
                skip(r, "narrow/block-scaled MMA requires this suite's sm120a-specific binary"); return;
            }
        } else if constexpr (fp8_format(A)) {
            if (ctx.prop.major * 10 + ctx.prop.minor < 89) { skip(r, "FP8 MMA requires sm89 or newer; request emulated backend on sm80"); return; }
        }
    }
    const auto accumulator = ctx.opts.get("accumulator", integer_format(A) ? "i32" : (A == tensor_format::f64 ? "f64" : "f32"));
    if (accumulator != (integer_format(A) ? "i32" : (A == tensor_format::f64 ? "f64" : "f32"))) {
        skip(r, "requested accumulator is not implemented by this tensor microkernel", "not_implemented"); return;
    }
    const int64_t requested_exponent_a = ctx.opts.integer("scale-exp-a", 0), requested_exponent_b = ctx.opts.integer("scale-exp-b", 0);
    if (requested_exponent_a < -6 || requested_exponent_a > 6 || requested_exponent_b < -6 || requested_exponent_b > 6)
        throw std::invalid_argument("scale exponents must be in -6..6");
    const int exponent_a = int(requested_exponent_a), exponent_b = int(requested_exponent_b);
    if (Scale == 0 && (exponent_a != 0 || exponent_b != 0)) throw std::invalid_argument("nonunit scale exponents require mx/nv scaling");
    const float scale_product = std::ldexp(1.0f, exponent_a + exponent_b);
    const uint32_t sa = uint32_t(Scale == 3 ? 0x38 + 8 * exponent_a : 127 + exponent_a) * 0x01010101u;
    const uint32_t sb = uint32_t(Scale == 3 ? 0x38 + 8 * exponent_b : 127 + exponent_b) * 0x01010101u;
    const uint32_t a = packed_positive_one(A, Scale);
    const uint32_t b = A == tensor_format::b1_xor ? 0 : packed_positive_one(B, Scale);
    ctx.memory_guard(config.lanes() * (Ilp * count * sizeof(Accumulator) + sizeof(unsigned long long)));
    device_buffer<Accumulator> output(config.lanes() * Ilp * count);
    device_buffer<unsigned long long> clocks(config.lanes());
    r.samples_ms = ctx.measure([&] { tensor_kernel<A, B, Scale, Emulated, Ilp><<<config.blocks, config.threads, 0, ctx.stream>>>(output.get(), clocks.get(), config.iterations, reset_every, a, b, sa, sb, scale_product, alternate); });
    auto actual = copy_output(output, ctx.stream);
    r.validation = "passed";
    bool all_finite = true;
    for (size_t q = 0; q < actual.size(); ++q) {
        const size_t lane = (q / count) / Ilp;
        const double sign = alternate && ((lane / 32) & 1) ? -1 : 1;
        const double expected = sign * double(k) * final_iterations * scale_product;
        if constexpr (integer_format(A)) {
            const auto expected_bits = uint32_t(int64_t(expected));
            validate_integer_bits(r, actual[q], expected_bits, "uniform-fragment integer MMA mismatch");
        } else {
            all_finite = all_finite && std::isfinite(double(actual[q]));
            validate_number(r, double(actual[q]), expected, 0, 0);
        }
    }
    record_micro(r, config, clocks, ctx.stream);
    record_tensor_work(r, config, m, n, k, reset_every, binary_format(A), Emulated);
    r.unit = binary_format(A) ? "Gbit-pair/s" : (integer_format(A) ? "Gop/s" : "Gflop/s");
    r.metadata["work_domain"] = binary_format(A) ? "bit" : (integer_format(A) ? "integer" : "floating_point");
    r.execution = Emulated ? "uniform_fragment_scalar_emulation" : "requested_tensor_ptx";
    record_tensor_codegen<A, Emulated>(r);
    r.metadata["mma_shape"] = std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);
    r.metadata["validation_scope"] = "every output register, uniform +1/-1 matrix fragments and power-of-two scales; not a randomized matrix-layout conformance test";
    r.metadata["timing_scope"] = "register-resident repeated warp MMA; no global matrix loads, packing, quantization, or dequantization in timed region";
    r.metadata["scale_layout"] = Scale == 0 ? "none" : (Scale == 3 ? "E4M3, 16 elements per scale (NVFP4)" : "E8M0, 32 elements per scale (MX)");
    if constexpr (Emulated) r.metadata["emulation_limitation"] = "uniform decoded operands; useful scalar arithmetic baseline, not an optimized low-precision GEMM/dequantization implementation";
    r.metrics["scale_product"] = scale_product;
    r.metadata["reference_accumulator_finite"] = "true";
    r.metadata["accumulator_finite"] = all_finite ? "true" : "false";
}

template <tensor_format A, tensor_format B, bool Emulated, int Ilp>
__global__ void tensor_half_kernel(uint32_t* output, unsigned long long* clocks, int iterations, int reset_every,
                                   uint32_t packed_a, uint32_t packed_b, bool alternate_sign) {
    constexpr int k = tensor_k(A, 0);
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool negative = alternate_sign && ((lane / 32) & 1);
    const uint32_t a = negative ? negate_packed_one(packed_a, A, 0) : packed_a;
    const uint32_t b = packed_b;
    uint32_t values[Ilp][2] = {};
    const auto begin = clock64();
    for (int completed = 0; completed < iterations;) {
        if (completed) {
            #pragma unroll
            for (int j = 0; j < Ilp; ++j) values[j][0] = values[j][1] = 0;
        }
        const int remaining = iterations - completed;
        const int segment = reset_every && reset_every < remaining ? reset_every : remaining;
        #pragma unroll 1
        for (int it = 0; it < segment; ++it) {
            #pragma unroll
            for (int j = 0; j < Ilp; ++j) {
                if constexpr (!Emulated) {
                    if constexpr (A == tensor_format::f16) {
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                            "{%0,%1}, {%2,%2,%2,%2}, {%3,%3}, {%0,%1};"
                            : "+r"(values[j][0]), "+r"(values[j][1]) : "r"(a), "r"(b));
                    } else if constexpr (fp8_format(A) && fp8_format(B)) fp8_mma<A, B>::step_half(values[j], a, b);
                    else narrow_mma<A, B, false>::step_half(values[j], a, b);
                } else {
                    #pragma unroll
                    for (int word = 0; word < 2; ++word) {
                        uint32_t packed_result = 0;
                        #pragma unroll
                        for (int part = 0; part < 2; ++part) {
                            float acc = __half2float(__ushort_as_half(static_cast<unsigned short>((values[j][word] >> (part * 16)) & 0xffffu)));
                            #pragma unroll 1
                            for (int q = 0; q < k; ++q)
                                acc = floating_device<float, float_op::fma>(negative ? -1.0f : 1.0f, 1.0f, acc);
                            packed_result |= uint32_t(__half_as_ushort(__float2half_rn(acc))) << (part * 16);
                        }
                        values[j][word] = packed_result;
                    }
                }
            }
        }
        completed += segment;
    }
    const auto end = clock64();
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) {
        output[(lane * Ilp + j) * 2] = values[j][0];
        output[(lane * Ilp + j) * 2 + 1] = values[j][1];
    }
    clocks[lane] = end - begin;
}

template <tensor_format A, tensor_format B, bool Emulated, int Ilp>
void tensor_half_case(context& ctx, result& r, const micro_config& config) {
    if constexpr (!Emulated && fp8_format(A) && fp8_format(B)) {
        if (ctx.prop.major * 10 + ctx.prop.minor < 89) {
            skip(r, "FP8 with FP16 accumulator requires sm89 or newer; request the emulated backend on sm80"); return;
        }
    } else if constexpr (!Emulated && A != tensor_format::f16) {
        bool suffix_a = false;
        #ifdef ARCHBENCH_ARCH_SUFFIX_A
        suffix_a = true;
        #endif
        if (ctx.prop.major != 12 || ctx.prop.minor != 0 || !suffix_a) {
            skip(r, "narrow f16-accumulator MMA is implemented in the sm120a worker; use emulated backend otherwise"); return;
        }
    }
    const auto pattern = ctx.opts.get("pattern", "ones");
    if (pattern != "ones" && pattern != "alternating_sign") throw std::invalid_argument("tensor pattern must be ones or alternating_sign");
    if (ctx.opts.integer("scale-exp-a", 0) != 0 || ctx.opts.integer("scale-exp-b", 0) != 0)
        throw std::invalid_argument("half-accumulator tensor path is unscaled");
    const bool alternate = pattern == "alternating_sign";
    constexpr int k = tensor_k(A, 0);
    const int reset_every = tensor_model::reset_interval(ctx.opts.integer("reset-every", 0));
    const int final_iterations = tensor_model::final_segment_iterations(config.iterations, reset_every);
    const int longest_segment = tensor_model::largest_segment_iterations(config.iterations, reset_every);
    __half expected = __float2half_rn(0.0f), peak = expected;
    bool stagnated = false;
    for (int it = 0; it < longest_segment; ++it) {
        const __half previous = peak;
        peak = __float2half_rn(__half2float(peak) + float(k));
        if (__half_as_ushort(peak) == __half_as_ushort(previous)) stagnated = true;
        if (it + 1 == final_iterations) expected = peak;
    }
    r.metadata["accumulator_finite"] = std::isfinite(__half2float(peak)) ? "true" : "false";
    r.metadata["accumulator_stagnated"] = stagnated ? "true" : "false";
    if (!std::isfinite(__half2float(peak))) {
        skip(r, "FP16 accumulator reference overflows within a segment; use reset-every=512 for a finite throughput baseline", "not_validated");
        return;
    }
    ctx.memory_guard(config.lanes() * (Ilp * 2 * sizeof(uint32_t) + sizeof(unsigned long long)));
    device_buffer<uint32_t> output(config.lanes() * Ilp * 2);
    device_buffer<unsigned long long> clocks(config.lanes());
    r.samples_ms = ctx.measure([&] { tensor_half_kernel<A, B, Emulated, Ilp><<<config.blocks, config.threads, 0, ctx.stream>>>(output.get(), clocks.get(), config.iterations, reset_every, packed_positive_one(A, 0), packed_positive_one(B, 0), alternate); });
    auto actual = copy_output(output, ctx.stream);
    const uint16_t positive_bits = __half_as_ushort(expected);
    r.validation = "passed";
    bool all_finite = true;
    for (size_t q = 0; q < actual.size(); ++q) {
        const size_t lane = (q / 2) / Ilp;
        const bool negative = alternate && ((lane / 32) & 1);
        const uint32_t bits = positive_bits | (negative ? 0x8000u : 0u);
        const uint32_t wanted = bits | (bits << 16);
        if (actual[q] != wanted) {
            for (int part = 0; part < 2; ++part) {
                const auto observed = __half2float(__ushort_as_half(static_cast<unsigned short>(actual[q] >> (16 * part))));
                const auto reference = __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
                all_finite = all_finite && std::isfinite(observed);
                validate_number(r, observed, reference, 0, 0);
            }
            r.metrics["bit_pattern_mismatches"] += 1;
            r.validation = "failed"; r.status = "validation_failed"; r.reason = "f16 tensor accumulator bit-pattern mismatch";
        }
    }
    r.metadata["reference_accumulator_finite"] = "true";
    r.metadata["accumulator_finite"] = all_finite ? "true" : "false";
    record_micro(r, config, clocks, ctx.stream);
    record_tensor_work(r, config, 16, 8, k, reset_every, false, Emulated);
    r.unit = "Gflop/s";
    r.metadata["work_domain"] = "floating_point";
    r.execution = Emulated ? "uniform_fragment_scalar_emulation" : "requested_tensor_ptx";
    record_tensor_codegen<A, Emulated>(r);
    r.metadata["mma_shape"] = "16x8x" + std::to_string(k);
    r.metadata["accumulator"] = "f16";
    r.metadata["validation_scope"] = "every accumulator bit pattern; uniform signed-one fragments; host rounds once per MMA step";
    r.metadata["timing_scope"] = "register-resident MMA, packing/quantization/global matrix loads excluded";
    r.metadata["emulation_accumulation"] = "emulated path uses f32 scalar dot products, rounded to f16 at each MMA boundary";
}

template <tensor_format A, tensor_format B, int Scale>
void dispatch_tensor_backend(context& ctx, result& r, const micro_config& config) {
    const auto backend = ctx.opts.get("backend", "native");
    if (ctx.opts.get("accumulator") == "f16") {
        if constexpr (Scale == 0 && (A == tensor_format::f16 || narrow_format(A))) {
            if (backend == "native" || backend == "ptx") { ARCHBENCH_ILP_DISPATCH(tensor_half_case, A, B, false); return; }
            if (backend == "emulated" || backend == "software") { ARCHBENCH_ILP_DISPATCH(tensor_half_case, A, B, true); return; }
        }
        skip(r, "f16 tensor accumulator is implemented for unscaled f16 and narrow floating pairs only"); return;
    }
    if (backend == "native" || backend == "ptx") { ARCHBENCH_ILP_DISPATCH(tensor_case, A, B, Scale, false); return; }
    if (backend == "emulated" || backend == "software") { ARCHBENCH_ILP_DISPATCH(tensor_case, A, B, Scale, true); return; }
    skip(r, "tensor backend must be native/ptx or emulated/software");
}

template <tensor_format A, tensor_format B>
void dispatch_tensor_scale(context& ctx, result& r, const micro_config& config) {
    const auto scale = ctx.opts.get("scale", "none");
    if (scale == "none") { dispatch_tensor_backend<A, B, 0>(ctx, r, config); return; }
    if constexpr (narrow_format(A) && narrow_format(B)) {
        if (scale == "mx") {
            if constexpr (A == tensor_format::e2m1 && B == tensor_format::e2m1) {
                if (ctx.opts.get("mma-kind", "auto") != "expanded") { dispatch_tensor_backend<A, B, 2>(ctx, r, config); return; }
            }
            dispatch_tensor_backend<A, B, 1>(ctx, r, config); return;
        }
        if constexpr (A == tensor_format::e2m1 && B == tensor_format::e2m1) {
            if (scale == "nv") { dispatch_tensor_backend<A, B, 3>(ctx, r, config); return; }
        }
    }
    skip(r, "this A/B type combination does not implement the requested scale mode");
}

template <tensor_format A>
void dispatch_tensor_b(context& ctx, result& r, const micro_config& config) {
    const auto dtype_b = ctx.opts.get("dtype-b", ctx.opts.get("dtype", "f16"));
    if constexpr (narrow_format(A)) {
        #define TENSOR_NARROW_B(NAME, ENUM) if (dtype_b == NAME) { dispatch_tensor_scale<A, tensor_format::ENUM>(ctx, r, config); return; }
        TENSOR_NARROW_B("e2m1", e2m1) TENSOR_NARROW_B("e2m3", e2m3) TENSOR_NARROW_B("e3m2", e3m2)
        TENSOR_NARROW_B("e4m3", e4m3) TENSOR_NARROW_B("e5m2", e5m2)
        #undef TENSOR_NARROW_B
    } else if constexpr (A == tensor_format::i8 || A == tensor_format::u8) {
        if (dtype_b == "i8") { dispatch_tensor_scale<A, tensor_format::i8>(ctx, r, config); return; }
        if (dtype_b == "u8") { dispatch_tensor_scale<A, tensor_format::u8>(ctx, r, config); return; }
    } else if constexpr (A == tensor_format::i4 || A == tensor_format::u4) {
        if (dtype_b == "i4") { dispatch_tensor_scale<A, tensor_format::i4>(ctx, r, config); return; }
        if (dtype_b == "u4") { dispatch_tensor_scale<A, tensor_format::u4>(ctx, r, config); return; }
    } else if (dtype_b == ctx.opts.get("dtype", "f16")) {
        dispatch_tensor_scale<A, A>(ctx, r, config); return;
    }
    skip(r, "unsupported tensor operand type pairing");
}

void dispatch_tensor(context& ctx, result& r, const micro_config& config) {
    if (ctx.opts.get("op", "mma") != "mma") { skip(r, "only dense mma is handled by this instruction microbenchmark", "not_implemented"); return; }
    const auto dtype = ctx.opts.get("dtype", "f16");
    #define TENSOR_A(NAME, ENUM) if (dtype == NAME) { dispatch_tensor_b<tensor_format::ENUM>(ctx, r, config); return; }
    TENSOR_A("f16", f16) TENSOR_A("bf16", bf16) TENSOR_A("tf32", tf32) TENSOR_A("f64", f64)
    TENSOR_A("i8", i8) TENSOR_A("u8", u8) TENSOR_A("i4", i4) TENSOR_A("u4", u4)
    TENSOR_A("b1_and", b1_and) TENSOR_A("b1_xor", b1_xor)
    TENSOR_A("e2m1", e2m1) TENSOR_A("e2m3", e2m3) TENSOR_A("e3m2", e3m2) TENSOR_A("e4m3", e4m3) TENSOR_A("e5m2", e5m2)
    #undef TENSOR_A
    skip(r, "tensor dtype not implemented", "not_implemented");
}

// Arbitrary-width integer arithmetic uses two 64-bit limbs and an explicit mask.
// The semantics are defined modulo 2^width, including MIN/-1 signed division.
struct limb_integer { uint64_t lo, hi; };

__host__ __device__ limb_integer limb_mask(limb_integer x, unsigned width) {
    if (width < 64) { x.lo &= (uint64_t(1) << width) - 1; x.hi = 0; }
    else if (width == 64) x.hi = 0;
    else if (width < 128) x.hi &= (uint64_t(1) << (width - 64)) - 1;
    return x;
}
__host__ __device__ bool limb_less(limb_integer a, limb_integer b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
__host__ __device__ bool limb_zero(limb_integer a) { return a.lo == 0 && a.hi == 0; }
__host__ __device__ bool limb_bit(limb_integer a, unsigned bit) { return bit < 64 ? (a.lo >> bit) & 1 : (a.hi >> (bit - 64)) & 1; }
__host__ __device__ limb_integer limb_add(limb_integer a, limb_integer b) {
    limb_integer c{a.lo + b.lo, a.hi + b.hi}; c.hi += c.lo < a.lo; return c;
}
__host__ __device__ limb_integer limb_negate(limb_integer a) { return limb_add({~a.lo, ~a.hi}, {1, 0}); }
__host__ __device__ limb_integer limb_sub(limb_integer a, limb_integer b) { return limb_add(a, limb_negate(b)); }
__host__ __device__ limb_integer limb_shl(limb_integer a, unsigned shift) {
    if (!shift) return a;
    if (shift >= 128) return {0, 0};
    if (shift >= 64) return {0, a.lo << (shift - 64)};
    return {a.lo << shift, (a.hi << shift) | (a.lo >> (64 - shift))};
}
__host__ __device__ limb_integer limb_shr(limb_integer a, unsigned shift) {
    if (!shift) return a;
    if (shift >= 128) return {0, 0};
    if (shift >= 64) return {a.hi >> (shift - 64), 0};
    return {(a.lo >> shift) | (a.hi << (64 - shift)), a.hi >> shift};
}
__host__ __device__ limb_integer limb_seed(size_t lane, int chain, unsigned width) {
    return limb_mask({integer_seed<uint64_t>(lane, chain), integer_seed<uint64_t>(lane + 0x1234, chain + 3)}, width);
}
__device__ __forceinline__ limb_integer limb_mul(limb_integer a, limb_integer b) {
    return {a.lo * b.lo, __umul64hi(a.lo, b.lo) + a.hi * b.lo + a.lo * b.hi};
}

__device__ __forceinline__ limb_integer limb_divide(limb_integer a, limb_integer b, limb_integer& remainder, unsigned width) {
    limb_integer quotient{0, 0}; remainder = {0, 0};
    #pragma unroll 1
    for (int bit = int(width) - 1; bit >= 0; --bit) {
        const bool carry = (remainder.hi >> 63) != 0;
        remainder = limb_shl(remainder, 1);
        remainder.lo |= limb_bit(a, unsigned(bit));
        if (carry || !limb_less(remainder, b)) {
            remainder = limb_sub(remainder, b);
            if (bit < 64) quotient.lo |= uint64_t(1) << bit;
            else quotient.hi |= uint64_t(1) << (bit - 64);
        }
    }
    return quotient;
}

template <int_op Op>
__device__ __forceinline__ limb_integer limb_device(limb_integer a, limb_integer b, limb_integer c, unsigned width, bool is_signed) {
    limb_integer out{0, 0};
    const bool negative_a = is_signed && limb_bit(a, width - 1);
    const bool negative_b = is_signed && limb_bit(b, width - 1);
    if constexpr (Op == int_op::add) out = limb_add(a, b);
    if constexpr (Op == int_op::sub) out = limb_sub(a, b);
    if constexpr (Op == int_op::mul) out = limb_mul(a, b);
    if constexpr (Op == int_op::mad) out = limb_add(limb_mul(a, b), c);
    if constexpr (Op == int_op::div || Op == int_op::rem) {
        const limb_integer magnitude_a = negative_a ? limb_mask(limb_negate(a), width) : a;
        const limb_integer magnitude_b = negative_b ? limb_mask(limb_negate(b), width) : b;
        limb_integer remainder;
        const limb_integer quotient = limb_divide(magnitude_a, magnitude_b, remainder, width);
        if constexpr (Op == int_op::div) out = negative_a != negative_b ? limb_negate(quotient) : quotient;
        else out = negative_a ? limb_negate(remainder) : remainder;
    }
    if constexpr (Op == int_op::min || Op == int_op::max) {
        const bool less = negative_a != negative_b ? negative_a : limb_less(a, b);
        if constexpr (Op == int_op::min) out = less ? a : b;
        else out = less ? b : a;
    }
    if constexpr (Op == int_op::bit_and) out = {a.lo & b.lo, a.hi & b.hi};
    if constexpr (Op == int_op::bit_or) out = {a.lo | b.lo, a.hi | b.hi};
    if constexpr (Op == int_op::bit_xor) out = {a.lo ^ b.lo, a.hi ^ b.hi};
    if constexpr (Op == int_op::shl) out = limb_shl(a, unsigned(b.lo % width));
    if constexpr (Op == int_op::shr) {
        const unsigned shift = unsigned(b.lo % width);
        out = limb_shr(a, shift);
        if (negative_a && shift) {
            const auto sign_fill = limb_shl(limb_mask({UINT64_MAX, UINT64_MAX}, width), width - shift);
            out.lo |= sign_fill.lo; out.hi |= sign_fill.hi;
        }
    }
    // Make both limbs observable to the optimizer across every recurrence step.
    asm volatile("mov.b64 %0, %1;" : "=l"(out.lo) : "l"(out.lo));
    asm volatile("mov.b64 %0, %1;" : "=l"(out.hi) : "l"(out.hi));
    return limb_mask(out, width);
}

#if defined(__SIZEOF_INT128__)
using host_uint128 = unsigned __int128;
host_uint128 host_value(limb_integer x) { return (host_uint128(x.hi) << 64) | x.lo; }
limb_integer host_limbs(host_uint128 x) { return {uint64_t(x), uint64_t(x >> 64)}; }

template <int_op Op>
limb_integer limb_reference(limb_integer aa, limb_integer bb, limb_integer cc, unsigned width, bool is_signed) {
    const host_uint128 mask = width == 128 ? ~host_uint128(0) : (host_uint128(1) << width) - 1;
    const host_uint128 a = host_value(aa), b = host_value(bb), c = host_value(cc);
    const bool negative_a = is_signed && ((a >> (width - 1)) & 1);
    const bool negative_b = is_signed && ((b >> (width - 1)) & 1);
    host_uint128 out = 0;
    if constexpr (Op == int_op::add) out = a + b;
    if constexpr (Op == int_op::sub) out = a - b;
    if constexpr (Op == int_op::mul) out = a * b;
    if constexpr (Op == int_op::mad) out = a * b + c;
    if constexpr (Op == int_op::div || Op == int_op::rem) {
        const auto ma = negative_a ? (-a & mask) : a;
        const auto mb = negative_b ? (-b & mask) : b;
        if constexpr (Op == int_op::div) { out = ma / mb; if (negative_a != negative_b) out = -out; }
        else { out = ma % mb; if (negative_a) out = -out; }
    }
    if constexpr (Op == int_op::min || Op == int_op::max) {
        const bool less = negative_a != negative_b ? negative_a : a < b;
        if constexpr (Op == int_op::min) out = less ? a : b;
        else out = less ? b : a;
    }
    if constexpr (Op == int_op::bit_and) out = a & b;
    if constexpr (Op == int_op::bit_or) out = a | b;
    if constexpr (Op == int_op::bit_xor) out = a ^ b;
    const unsigned shift = unsigned(bb.lo % width);
    if constexpr (Op == int_op::shl) out = a << shift;
    if constexpr (Op == int_op::shr) {
        out = a >> shift;
        if (negative_a && shift) out |= mask << (width - shift);
    }
    return host_limbs(out & mask);
}
#endif

template <int_op Op, int Ilp>
__global__ void limb_kernel(limb_integer* output, unsigned long long* clocks, int iterations,
                           limb_integer b, limb_integer c, unsigned width, bool is_signed) {
    const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    limb_integer values[Ilp];
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) values[j] = limb_seed(lane, j, width);
    const auto begin = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int j = 0; j < Ilp; ++j) values[j] = limb_device<Op>(values[j], b, c, width, is_signed);
    }
    const auto end = clock64();
    #pragma unroll
    for (int j = 0; j < Ilp; ++j) output[lane * Ilp + j] = values[j];
    clocks[lane] = end - begin;
}

template <int_op Op, int Ilp>
void limb_case(context& ctx, result& r, const micro_config& config) {
#if defined(__SIZEOF_INT128__)
    const auto dtype = ctx.opts.get("dtype");
    const unsigned width = unsigned(std::stoul(dtype.substr(1)));
    const bool is_signed = dtype[0] == 'i';
    const int64_t operand = ctx.opts.integer("operand", 13), addend = ctx.opts.integer("addend", 17);
    auto b = limb_mask({uint64_t(operand), operand < 0 ? UINT64_MAX : 0}, width);
    const auto c = limb_mask({uint64_t(addend), addend < 0 ? UINT64_MAX : 0}, width);
    if constexpr (Op == int_op::div || Op == int_op::rem) {
        if (limb_zero(b)) throw std::invalid_argument("limb integer divisor is zero after width truncation");
    }
    ctx.memory_guard(config.lanes() * (Ilp * sizeof(limb_integer) + sizeof(unsigned long long)));
    device_buffer<limb_integer> output(config.lanes() * Ilp);
    device_buffer<unsigned long long> clocks(config.lanes());
    r.samples_ms = ctx.measure([&] { limb_kernel<Op, Ilp><<<config.blocks, config.threads, 0, ctx.stream>>>(output.get(), clocks.get(), config.iterations, b, c, width, is_signed); });
    auto actual = copy_output(output, ctx.stream);
    r.validation = "passed";
    for (size_t q : micro_sample_indices(config.lanes(), Ilp)) {
        auto expected = limb_seed(q / Ilp, int(q % Ilp), width);
        for (int i = 0; i < config.iterations; ++i) expected = limb_reference<Op>(expected, b, c, width, is_signed);
        validate_integer_bits(r, host_value(actual[q]), host_value(expected),
                              "two-limb GPU integer differs from host unsigned __int128 reference");
    }
    record_micro(r, config, clocks, ctx.stream, Op == int_op::mad ? 2 : 1);
    r.execution = "software_two_limb_integer";
    r.unit = "Gop/s";
    r.metadata["work_domain"] = "integer";
    r.metadata["integer_semantics"] = "two's complement modulo 2^width; truncating signed division; MIN/-1 wraps; shift count is low64(operand) modulo width";
    r.metadata["emulation_scope"] = "complete arithmetic on two u64 limbs, width mask applied after every operation; not a native-width throughput claim";
    r.metrics["integer_width"] = width;
#else
    skip(r, "nonstandard-width integer validation requires host compiler unsigned __int128 support");
#endif
}

void dispatch_limb(context& ctx, result& r, const micro_config& config) {
    const auto op = ctx.opts.get("op", "add");
    #define LIMB_CASE(NAME, ENUM) if (op == NAME) { ARCHBENCH_ILP_DISPATCH(limb_case, int_op::ENUM); return; }
    LIMB_CASE("add", add) LIMB_CASE("sub", sub) LIMB_CASE("mul", mul) LIMB_CASE("mad", mad)
    LIMB_CASE("div", div) LIMB_CASE("rem", rem) LIMB_CASE("min", min) LIMB_CASE("max", max)
    LIMB_CASE("and", bit_and) LIMB_CASE("or", bit_or) LIMB_CASE("xor", bit_xor) LIMB_CASE("shl", shl) LIMB_CASE("shr", shr)
    #undef LIMB_CASE
    skip(r, "nonstandard-width integer operation not implemented", "not_implemented");
}

#undef ARCHBENCH_ILP_DISPATCH

}  // namespace

bool run_alu(context& ctx, result& r) {
    const auto family = ctx.opts.get("family");
    const auto dtype = ctx.opts.get("dtype", "f32");
    if (family == "convert") {
        if (ctx.opts.get("op", "cast") != "cast") { skip(r, "only typed cast conversion is implemented", "not_implemented"); return true; }
        #define CONVERT_SOURCE(NAME, TYPE) if (dtype == NAME) { dispatch_conversion_destination<TYPE>(ctx, r); return true; }
        CONVERT_SOURCE("i8", int8_t) CONVERT_SOURCE("u8", uint8_t) CONVERT_SOURCE("i16", int16_t) CONVERT_SOURCE("u16", uint16_t)
        CONVERT_SOURCE("i32", int32_t) CONVERT_SOURCE("u32", uint32_t) CONVERT_SOURCE("i64", int64_t) CONVERT_SOURCE("u64", uint64_t)
        CONVERT_SOURCE("f16", __half) CONVERT_SOURCE("bf16", __nv_bfloat16) CONVERT_SOURCE("f32", float) CONVERT_SOURCE("f64", double)
        #undef CONVERT_SOURCE
        skip(r, "conversion source is not implemented", "not_implemented"); return true;
    }
    if (family != "integer" && family != "scalar" && family != "bit" && family != "packed" && family != "sfu" && family != "tensor") return false;
    if (family != "tensor" && !ctx.opts.get("reset-every").empty())
        throw std::invalid_argument("reset-every is implemented for tensor MMA microbenchmarks only");
    const auto config = configuration(ctx);
    r.metrics["grid_blocks_per_sm"] = double(config.blocks) / ctx.prop.multiProcessorCount;
    if (family == "integer") {
        #define INTEGER_TYPE(NAME, TYPE) if (dtype == NAME) { dispatch_integer<TYPE>(ctx, r, config); return true; }
        INTEGER_TYPE("i8", int8_t) INTEGER_TYPE("u8", uint8_t) INTEGER_TYPE("i16", int16_t) INTEGER_TYPE("u16", uint16_t)
        INTEGER_TYPE("i32", int32_t) INTEGER_TYPE("u32", uint32_t) INTEGER_TYPE("i64", int64_t) INTEGER_TYPE("u64", uint64_t)
        #undef INTEGER_TYPE
        if (dtype.size() > 1 && (dtype[0] == 'i' || dtype[0] == 'u') &&
            std::all_of(dtype.begin() + 1, dtype.end(), [](char ch) { return ch >= '0' && ch <= '9'; })) {
            const auto width = std::stoul(dtype.substr(1));
            if (width >= 1 && width <= 128) { dispatch_limb(ctx, r, config); return true; }
        }
    } else if (family == "scalar") {
        if (dtype == "f16") { dispatch_floating<__half>(ctx, r, config); return true; }
        if (dtype == "bf16") { dispatch_floating<__nv_bfloat16>(ctx, r, config); return true; }
        if (dtype == "f32") { dispatch_floating<float>(ctx, r, config); return true; }
        if (dtype == "f64") { dispatch_floating<double>(ctx, r, config); return true; }
    } else if (family == "bit") {
        if (dtype == "u32") { dispatch_bit<uint32_t>(ctx, r, config); return true; }
        if (dtype == "u64") { dispatch_bit<uint64_t>(ctx, r, config); return true; }
    } else if (family == "sfu") {
        if (dtype != "f32") { skip(r, "SFU source implements f32 only", "not_implemented"); return true; }
        const auto precision = ctx.opts.get("precision", "approx");
        if (precision == "approx") dispatch_sfu<true>(ctx, r, config);
        else if (precision == "precise") dispatch_sfu<false>(ctx, r, config);
        else throw std::invalid_argument("precision must be approx or precise");
        return true;
    } else if (family == "packed") {
        if (dtype == "f16x2") { dispatch_packed_float<__half>(ctx, r, config); return true; }
        if (dtype == "bf16x2") { dispatch_packed_float<__nv_bfloat16>(ctx, r, config); return true; }
        const auto dtype_b = ctx.opts.get("dtype-b", dtype == "u8" || dtype == "u16" ? "u8" : "i8");
        if ((dtype == "i8" || dtype == "u8" || dtype == "i16" || dtype == "u16") && (dtype_b == "i8" || dtype_b == "u8")) {
            const bool sa = dtype[0] == 'i', sb = dtype_b[0] == 'i';
            const auto op = ctx.opts.get("op");
            if ((op == "dp4a" && dtype != "i8" && dtype != "u8") ||
                ((op == "dp2a_lo" || op == "dp2a_hi") && dtype != "i16" && dtype != "u16")) {
                skip(r, "DP4A uses byte A; DP2A uses 16-bit A"); return true;
            }
            if (sa && sb) dispatch_dot_kind<true, true>(ctx, r, config);
            else if (sa) dispatch_dot_kind<true, false>(ctx, r, config);
            else if (sb) dispatch_dot_kind<false, true>(ctx, r, config);
            else dispatch_dot_kind<false, false>(ctx, r, config);
            return true;
        }
    } else if (family == "tensor") { dispatch_tensor(ctx, r, config); return true; }
    skip(r, "dtype not implemented for the selected arithmetic family", "not_implemented");
    return true;
}

}  // namespace archbench

