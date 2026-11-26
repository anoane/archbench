// SPDX-License-Identifier: AGPL-3.0-or-later
// Systems, memory, synchronization, library primitives, and composite workloads.
#include "archbench/common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(ARCHBENCH_HAS_CUSPARSE) && ARCHBENCH_HAS_CUSPARSE
#include <cusparse.h>
#endif
#if defined(ARCHBENCH_HAS_CUFFT) && ARCHBENCH_HAS_CUFFT
#include <cufft.h>
#include <cufftXt.h>
#endif

namespace archbench {
namespace {

using wall_clock = std::chrono::steady_clock;

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("size product overflow");
    return a * b;
}

std::size_t positive_size(const options& opts, const std::string& key,
                          std::int64_t fallback) {
    const auto v = opts.integer(key, fallback);
    if (v <= 0) throw std::invalid_argument(key + " must be positive");
    return static_cast<std::size_t>(v);
}

int thread_count(const context& ctx, bool power_of_two = false) {
    const auto v = ctx.opts.integer("threads", 256);
    if (v < 32 || v > ctx.prop.maxThreadsPerBlock || v % 32 != 0 ||
        (power_of_two && (v & (v - 1))))
        throw std::invalid_argument("threads must be a supported multiple of 32"
                                    " (and a power of two for row reductions)");
    return static_cast<int>(v);
}

int grid_count(const context& ctx, std::size_t n, int threads) {
    const auto fallback = std::max(1, ctx.prop.multiProcessorCount * 4);
    const auto requested = ctx.opts.integer("blocks", fallback);
    if (requested < 1 || requested > ctx.prop.maxGridSize[0])
        throw std::invalid_argument("blocks outside the device grid limit");
    const auto needed = std::max<std::size_t>(1, (n + threads - 1) / threads);
    return static_cast<int>(std::min<std::size_t>(requested, needed));
}

// A sample averages completed host-clock calls. Preparation runs before EVERY
// call, outside its measured interval, so managed migration never degenerates
// into repeated access to an already resident allocation.
std::vector<double> measure_host_completed(context& ctx,
        const std::function<void()>& prepare, const std::function<void()>& launch) {
    const auto policy = ctx.measurement_policy();
    ctx.timing_metrics.clear();
    ctx.timing_metadata.clear();
    auto milliseconds = [](wall_clock::time_point start) {
        return std::chrono::duration<double, std::milli>(wall_clock::now() - start).count();
    };
    double preparation_ms = 0;
    auto one = [&] {
        const auto preparation_start = wall_clock::now();
        prepare();
        preparation_ms += milliseconds(preparation_start);
        const auto start = wall_clock::now();
        launch();
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        const double elapsed = milliseconds(start);
        if (!(elapsed > 0) || !std::isfinite(elapsed))
            throw std::runtime_error("nonpositive host timing interval");
        return elapsed;
    };
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const auto warmup_start = wall_clock::now();
    std::uint64_t warmup_calls = 0;
    while (warmup_calls < static_cast<std::uint64_t>(policy.warmup) ||
           milliseconds(warmup_start) < policy.warmup_ms) {
        one(); ++warmup_calls;
    }
    ctx.timing_metrics["warmup_host_ms"] = milliseconds(warmup_start);
    ctx.timing_metrics["warmup_preparation_host_ms"] = preparation_ms;
    ctx.timing_metrics["warmup_calls"] = static_cast<double>(warmup_calls);
    ctx.timing_metadata["warmup_scope"] = "host wall time including per-call preparation and synchronized completion";
    const auto calibration_start = wall_clock::now();
    const double preparation_before_calibration = preparation_ms;
    double calibration_ms = 0;
    ctx.last_repeats = policy.repeats;
    if (!ctx.last_repeats) {
        calibration_ms = one();
        ctx.last_repeats = static_cast<int>(std::max(1., std::min(
            static_cast<double>(policy.max_repeats), std::ceil(policy.target_ms / calibration_ms))));
    }
    ctx.timing_metrics["calibration_host_ms"] = milliseconds(calibration_start);
    ctx.timing_metrics["calibration_preparation_host_ms"] = preparation_ms - preparation_before_calibration;
    ctx.timing_metrics["calibration_timed_calls_host_ms"] = calibration_ms;
    ctx.timing_metrics["calibration_calls"] = policy.repeats ? 0 : 1;
    ctx.timing_metadata["repetition_policy"] = policy.repeats ? "fixed" : "automatic_target_duration";
    const auto measurement_start = wall_clock::now();
    const double preparation_before_measurement = preparation_ms;
    double measured_ms = 0;
    std::vector<double> samples;
    samples.reserve(policy.maximum_samples);
    while (policy.needs_more(samples)) {
        double elapsed = 0;
        for (int repeat = 0; repeat < ctx.last_repeats; ++repeat) elapsed += one();
        measured_ms += elapsed;
        samples.push_back(elapsed / ctx.last_repeats);
    }
    ctx.timing_metrics["measurement_host_ms"] = milliseconds(measurement_start);
    ctx.timing_metrics["measured_preparation_host_ms"] = preparation_ms - preparation_before_measurement;
    ctx.timing_metrics["measured_timed_calls_host_ms"] = measured_ms;
    ctx.timing_metadata["timing_sample_scope"] = "mean of individually synchronized host-clock calls; per-call preparation excluded";
    ctx.record_measurement(samples, "individual_host_completed_calls_excluding_preparation");
    return samples;
}

template <class T>
void upload(context& ctx, T* dst, const std::vector<T>& src) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src.data(), src.size() * sizeof(T),
                               cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
}

template <class T>
std::vector<T> download(context& ctx, const T* src, std::size_t n) {
    std::vector<T> dst(n);
    CUDA_CHECK(cudaMemcpyAsync(dst.data(), src, n * sizeof(T),
                               cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    return dst;
}

float input_value(std::size_t i, unsigned salt = 0) {
    // Exactly representable, nonconstant values, independent of platform RNG.
    const auto h = static_cast<std::uint32_t>(i) * 1664525u +
                   1013904223u + salt * 2246822519u;
    return static_cast<float>(static_cast<int>((h >> 16) % 257u) - 128) / 128.0f;
}

std::vector<float> float_input(std::size_t n, unsigned salt = 0) {
    std::vector<float> x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = input_value(i, salt);
    return x;
}

void check_value(result& r, long double got, long double expected,
                 long double atol = 0, long double rtol = 0) {
    if (!std::isfinite(got) || !std::isfinite(expected)) {
        r.validation = "failed";
        r.status = "validation_failed";
        r.reason = "nonfinite value encountered in output/reference validation";
        r.metrics["nonfinite_validation_values"] += 1;
        r.max_abs_error = r.max_rel_error = std::numeric_limits<double>::quiet_NaN();
        return;
    }
    const auto error = std::fabs(got - expected);
    const auto scale = std::fabs(expected);
    const auto relative = error / std::max(scale, 1.0e-30L);
    r.max_abs_error = std::max(r.max_abs_error, static_cast<double>(error));
    r.max_rel_error = std::max(r.max_rel_error, static_cast<double>(relative));
    if (error > atol + rtol * scale) {
        r.validation = "failed";
        r.status = "validation_failed";
        r.reason = "computed output disagrees with the independent CPU reference";
    }
}

void begin_validation(result& r, std::size_t checked) {
    r.validation = "passed";
    r.metadata["validation_scope"] = "all_output_elements";
    r.metrics["validated_elements"] = static_cast<double>(checked);
}

void check_launch() { CUDA_CHECK(cudaGetLastError()); }

enum class memory_op { copy, write, triad, strided, gather, scatter };

__global__ void streaming_kernel(const float* x, const float* z, float* y,
                                 const std::uint32_t* indices, std::size_t n,
                                 std::size_t stride, memory_op op) {
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x; i < n; i += step) {
        switch (op) {
        case memory_op::copy: y[i] = x[i]; break;
        case memory_op::write: y[i] = static_cast<float>(i & 255u); break;
        case memory_op::triad: y[i] = x[i] + 0.5f * z[i]; break;
        case memory_op::strided: y[i] = x[(i * stride) % n]; break;
        case memory_op::gather: y[i] = x[indices[i]]; break;
        case memory_op::scatter: y[indices[i]] = x[i]; break;
        }
    }
}

__global__ void streaming_read_kernel(const float* x, float* sums, std::size_t n) {
    const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    float sum = 0;
    for (std::size_t i = t; i < n; i += step) sum += x[i];
    sums[t] = sum;
}

__global__ void pointer_chase_kernel(const volatile std::uint32_t* next,
                                     unsigned long long* out, std::uint32_t start,
                                     unsigned iterations) {
    if (blockIdx.x || threadIdx.x) return;
    auto index = start;
    const auto t0 = clock64();
    for (unsigned k = 0; k < iterations; ++k) index = next[index];
    const auto t1 = clock64();
    out[0] = index;
    out[1] = t1 - t0;
}

__global__ void shared_bank_kernel(float* out, unsigned stride,
                                   unsigned iterations) {
    extern __shared__ float storage[];
    volatile float* values = storage;
    const unsigned offset = threadIdx.x * stride;
    values[offset] = static_cast<float>((threadIdx.x % 7) + 1);
    __syncthreads();
    float sum = 0;
    for (unsigned k = 0; k < iterations; ++k) sum += values[offset];
    out[static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = sum;
}

template <bool asynchronous>
__global__ void staged_copy_kernel(const float4* input, float4* output,
                                   std::size_t vectors) {
    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* shared = reinterpret_cast<float4*>(shared_raw);
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockDim.x;
         base < vectors; base += step) {
        const auto i = base + threadIdx.x;
        if (i < vectors) {
            if constexpr (asynchronous) {
#if __CUDA_ARCH__ >= 800
                const auto address = static_cast<unsigned>(
                    __cvta_generic_to_shared(shared + threadIdx.x));
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                             :: "r"(address), "l"(input + i) : "memory");
#else
                shared[threadIdx.x] = input[i];
#endif
            } else {
                shared[threadIdx.x] = input[i];
            }
        }
        if constexpr (asynchronous) {
#if __CUDA_ARCH__ >= 800
            asm volatile("cp.async.commit_group;\n"
                         "cp.async.wait_group 0;\n" ::: "memory");
#endif
        }
        __syncthreads();
        if (i < vectors) output[i] = shared[threadIdx.x];
        __syncthreads();
    }
}

bool run_memory(context& ctx, result& r) {
    const auto op = ctx.opts.get("op", "copy");
    const auto dtype = ctx.opts.get("dtype", op == "pointer_chase" ? "u32" : "f32");
    if ((op == "pointer_chase" && dtype != "u32") ||
        (op != "pointer_chase" && dtype != "f32")) {
        skip(r, "this memory kernel uses f32 elements (pointer_chase uses u32)");
        return true;
    }
    const auto n = positive_size(ctx.opts, "elements", 1 << 20);
    const int threads = thread_count(ctx);
    const int blocks = grid_count(ctx, n, threads);
    const auto thread_total = static_cast<std::size_t>(blocks) * threads;
    r.execution = "compiler_lowered";
    r.unit = "GB/s";
    r.metadata["work_domain"] = (op == "read" || op == "shared_bank" || op == "pointer_chase") ? "load" :
                                 op == "triad" ? "floating_point" : "data_movement";
    r.metadata["cache_state"] = "warm_repeated; working set does not prove cache residency";
    r.metrics["working_set_bytes"] = static_cast<double>(n * sizeof(float));
    r.metrics["threads"] = threads;
    r.metrics["blocks"] = blocks;

    if (op == "pointer_chase") {
        if (n > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("pointer chase requires at most UINT32_MAX elements");
        const auto iterations = positive_size(ctx.opts, "iterations", 4096);
        if (iterations > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("pointer chase iteration count too large");
        ctx.memory_guard(checked_product(n, sizeof(std::uint32_t)) + 16);
        std::vector<std::uint32_t> permutation(n), next(n);
        std::iota(permutation.begin(), permutation.end(), 0u);
        std::mt19937 rng(static_cast<unsigned>(ctx.opts.integer("seed", 1)));
        std::shuffle(permutation.begin(), permutation.end(), rng);
        for (std::size_t i = 0; i < n; ++i)
            next[permutation[i]] = permutation[(i + 1) % n];
        device_buffer<std::uint32_t> d_next(n);
        device_buffer<unsigned long long> d_out(2);
        upload(ctx, d_next.get(), next);
        auto launch = [&] {
            pointer_chase_kernel<<<1, 32, 0, ctx.stream>>>(
                d_next.get(), d_out.get(), permutation[0], static_cast<unsigned>(iterations));
            check_launch();
        };
        r.samples_ms = ctx.measure(launch);
        launch();
        const auto out = download(ctx, d_out.get(), 2);
        begin_validation(r, 1);
        check_value(r, out[0], permutation[iterations % n]);
        r.metrics["cycles_per_dependent_load"] =
            static_cast<double>(out[1]) / static_cast<double>(iterations);
        r.metrics["cycle_sample_iterations"] = static_cast<double>(iterations);
        r.metadata["cycle_metric_scope"] = "one post-timing clock64 sample; loop overhead included";
        r.metadata["dependency"] = "single serialized random Hamiltonian cycle";
        r.metrics["threads"] = 32;
        r.metrics["blocks"] = 1;
        r.metrics["active_lanes"] = 1;
        r.metrics["allocated_pointer_array_bytes"] = static_cast<double>(n * sizeof(std::uint32_t));
        r.metrics["visited_unique_nodes_per_call"] = static_cast<double>(std::min(n, iterations));
        r.metadata["working_set_scope"] = "allocated pointer array; each call starts at the same node and may visit only a prefix";
        r.work_per_iteration = static_cast<double>(iterations);
        r.bytes_per_iteration = static_cast<double>(iterations * sizeof(std::uint32_t));
        r.unit = "Gload/s";
        return true;
    }
    if (op == "shared_bank") {
        const auto stride = positive_size(ctx.opts, "stride", 1);
        const auto iterations = positive_size(ctx.opts, "iterations", 1024);
        if (stride > 32 || iterations > (1u << 20))
            throw std::invalid_argument("shared-bank stride must be <=32, iterations <=2^20");
        const auto shared_bytes = checked_product(threads * stride, sizeof(float));
        if (shared_bytes > ctx.prop.sharedMemPerBlock) {
            skip(r, "requested shared-memory bank layout exceeds per-block shared memory");
            return true;
        }
        ctx.memory_guard(thread_total * sizeof(float));
        device_buffer<float> out(thread_total);
        auto launch = [&] {
            shared_bank_kernel<<<blocks, threads, shared_bytes, ctx.stream>>>(
                out.get(), static_cast<unsigned>(stride), static_cast<unsigned>(iterations));
            check_launch();
        };
        r.samples_ms = ctx.measure(launch);
        const auto got = download(ctx, out.get(), thread_total);
        begin_validation(r, got.size());
        for (std::size_t i = 0; i < got.size(); ++i)
            check_value(r, got[i], ((i % threads) % 7 + 1) * iterations);
        r.bytes_per_iteration = static_cast<double>(thread_total * iterations * sizeof(float));
        r.work_per_iteration = static_cast<double>(thread_total * iterations);
        r.metrics["nominal_bank_conflict_degree"] = std::gcd<std::size_t>(stride, 32);
        r.metrics["working_set_bytes"] = static_cast<double>(shared_bytes);
        r.metrics["shared_allocation_bytes_per_block"] = static_cast<double>(shared_bytes);
        r.metrics["shared_read_addresses_per_block"] = threads;
        r.metadata["working_set_scope"] = "dynamic shared allocation per block, not the unrelated elements argument or DRAM footprint";
        r.metadata["address_pattern"] = "distinct per-lane addresses; no broadcast";
        r.metadata["bandwidth_scope"] = "requested shared read bytes, not DRAM traffic";
        return true;
    }

    if (n > std::numeric_limits<std::uint32_t>::max() &&
        (op == "gather" || op == "scatter"))
        throw std::invalid_argument("gather/scatter indices are u32");
    ctx.memory_guard(checked_product(n, 4 * sizeof(float)) + thread_total * sizeof(float));
    const auto x = float_input(n), z = float_input(n, 3);
    device_buffer<float> dx(n), dz(n), dy(std::max(n, thread_total));
    upload(ctx, dx.get(), x);
    upload(ctx, dz.get(), z);
    if (op == "cp_async" || op == "shared_copy") {
        if (n % 4) {
            skip(r, "16-byte staged-copy kernels require elements divisible by four");
            return true;
        }
        auto launch = [&] {
            if (op == "cp_async")
                staged_copy_kernel<true><<<blocks, threads, threads * sizeof(float4), ctx.stream>>>(
                    reinterpret_cast<const float4*>(dx.get()),
                    reinterpret_cast<float4*>(dy.get()), n / 4);
            else
                staged_copy_kernel<false><<<blocks, threads, threads * sizeof(float4), ctx.stream>>>(
                    reinterpret_cast<const float4*>(dx.get()),
                    reinterpret_cast<float4*>(dy.get()), n / 4);
            check_launch();
        };
        r.samples_ms = ctx.measure(launch);
        const auto got = download(ctx, dy.get(), n);
        begin_validation(r, n);
        for (std::size_t i = 0; i < n; ++i) check_value(r, got[i], x[i]);
        r.bytes_per_iteration = static_cast<double>(n * sizeof(float) * 2);
        r.execution = op == "cp_async" ? "requested_ptx" : "compiler_lowered";
        r.metadata["pipeline"] = "single shared-memory stage; cp.async commit/wait per tile";
        return true;
    }
    if (op == "read") {
        auto launch = [&] {
            streaming_read_kernel<<<blocks, threads, 0, ctx.stream>>>(dx.get(), dy.get(), n);
            check_launch();
        };
        r.samples_ms = ctx.measure(launch);
        const auto got = download(ctx, dy.get(), thread_total);
        begin_validation(r, thread_total);
        for (std::size_t t = 0; t < thread_total; ++t) {
            long double ref = 0;
            for (std::size_t i = t; i < n; i += thread_total) ref += x[i];
            check_value(r, got[t], ref, 1.0e-4L, 1.0e-5L);
        }
        r.bytes_per_iteration = static_cast<double>((n + thread_total) * sizeof(float));
        r.work_per_iteration = static_cast<double>(n);
        r.metadata["read_sink"] = "one partial sum store per CUDA thread";
        return true;
    }

    memory_op kind;
    if (op == "copy") kind = memory_op::copy;
    else if (op == "write") kind = memory_op::write;
    else if (op == "triad") kind = memory_op::triad;
    else if (op == "strided") kind = memory_op::strided;
    else if (op == "gather") kind = memory_op::gather;
    else if (op == "scatter") kind = memory_op::scatter;
    else { skip(r, "unknown memory operation: " + op); return true; }
    const auto stride = positive_size(ctx.opts, "stride", 1);
    if (stride > std::numeric_limits<std::size_t>::max() / n)
        throw std::invalid_argument("strided index multiplication would overflow");
    std::vector<std::uint32_t> indices;
    device_buffer<std::uint32_t> d_indices((op == "gather" || op == "scatter") ? n : 1);
    if (op == "gather" || op == "scatter") {
        indices.resize(n);
        std::iota(indices.begin(), indices.end(), 0u);
        std::mt19937 rng(static_cast<unsigned>(ctx.opts.integer("seed", 1)));
        std::shuffle(indices.begin(), indices.end(), rng);
        upload(ctx, d_indices.get(), indices);
        r.metadata["indices"] = "seeded random permutation; scatter has no data race";
    }
    auto launch = [&] {
        streaming_kernel<<<blocks, threads, 0, ctx.stream>>>(
            dx.get(), dz.get(), dy.get(), d_indices.get(), n, stride, kind);
        check_launch();
    };
    r.samples_ms = ctx.measure(launch);
    const auto got = download(ctx, dy.get(), n);
    begin_validation(r, n);
    for (std::size_t i = 0; i < n; ++i) {
        if (op == "scatter") check_value(r, got[indices[i]], x[i]);
        else {
            float ref = x[i];
            if (op == "write") ref = static_cast<float>(i & 255u);
            else if (op == "triad") ref = x[i] + 0.5f * z[i];
            else if (op == "strided") ref = x[(i * stride) % n];
            else if (op == "gather") ref = x[indices[i]];
            check_value(r, got[i], ref);
        }
    }
    const int streams = op == "write" ? 1 :
                        (op == "triad" || op == "gather" || op == "scatter" ? 3 : 2);
    r.bytes_per_iteration = static_cast<double>(n * sizeof(float) * streams);
    r.work_per_iteration = op == "triad" ? static_cast<double>(2 * n) : 0;
    r.metadata["bandwidth_scope"] = "requested global bytes including index/sink traffic; not physical transactions";
    return true;
}

enum class atomic_op { add, minimum, maximum, exchange, cas, bit_and, bit_or, bit_xor };

template <class T>
__device__ void integer_atomic(T* address, T value, atomic_op op) {
    if constexpr (sizeof(T) < 4) {
        // CAS on an aligned containing word preserves neighboring packed fields.
        // The allocation is padded to a complete word by atomic_case().
        const auto raw = reinterpret_cast<std::uintptr_t>(address);
        auto* word = reinterpret_cast<unsigned int*>(raw & ~std::uintptr_t(3));
        const unsigned shift = static_cast<unsigned>(raw & 3u) * 8u;
        constexpr unsigned field_mask = (1u << (sizeof(T) * 8)) - 1u;
        auto old = atomicCAS(word, 0u, 0u);
        unsigned assumed;
        do {
            assumed = old;
            const unsigned bits = (assumed >> shift) & field_mask;
            const T prior = static_cast<T>(bits);
            unsigned next = bits;
            switch (op) {
            case atomic_op::add: next = bits + static_cast<unsigned>(value); break;
            case atomic_op::minimum: next = static_cast<unsigned>(prior < value ? prior : value); break;
            case atomic_op::maximum: next = static_cast<unsigned>(prior > value ? prior : value); break;
            case atomic_op::exchange: next = 7u; break;
            case atomic_op::cas: next = bits == 0 ? 1u : bits; break;
            case atomic_op::bit_and: next = bits & 0x55u; break;
            case atomic_op::bit_or: next = bits | 0x55u; break;
            case atomic_op::bit_xor: next = bits ^ 0x55u; break;
            }
            const auto replacement = (assumed & ~(field_mask << shift)) | ((next & field_mask) << shift);
            old = atomicCAS(word, assumed, replacement);
        } while (old != assumed);
    } else if constexpr (sizeof(T) == 8) {
        auto* bits = reinterpret_cast<unsigned long long*>(address);
        switch (op) {
        case atomic_op::add: atomicAdd(bits, static_cast<unsigned long long>(value)); break;
        case atomic_op::minimum: atomicMin(address, value); break;
        case atomic_op::maximum: atomicMax(address, value); break;
        case atomic_op::exchange: atomicExch(bits, 7ull); break;
        case atomic_op::cas: atomicCAS(bits, 0ull, 1ull); break;
        case atomic_op::bit_and: atomicAnd(bits, 0x55ull); break;
        case atomic_op::bit_or: atomicOr(bits, 0x55ull); break;
        case atomic_op::bit_xor: atomicXor(bits, 0x55ull); break;
        }
    } else {
        auto* bits = reinterpret_cast<unsigned int*>(address);
        switch (op) {
        case atomic_op::add: atomicAdd(address, value); break;
        case atomic_op::minimum: atomicMin(address, value); break;
        case atomic_op::maximum: atomicMax(address, value); break;
        case atomic_op::exchange: atomicExch(bits, 7u); break;
        case atomic_op::cas: atomicCAS(bits, 0u, 1u); break;
        case atomic_op::bit_and: atomicAnd(bits, 0x55u); break;
        case atomic_op::bit_or: atomicOr(bits, 0x55u); break;
        case atomic_op::bit_xor: atomicXor(bits, 0x55u); break;
        }
    }
}

template <class T>
__device__ void float_atomic(T* address, T value, atomic_op op) {
    if (op == atomic_op::add) {
        atomicAdd(address, value);
        return;
    }
    // Float min/max/exchange/CAS use bitwise CAS, explicitly marked emulated.
    if constexpr (std::is_same<T, float>::value) {
        auto* bits = reinterpret_cast<unsigned int*>(address);
        if (op == atomic_op::exchange) { atomicExch(bits, __float_as_uint(7.0f)); return; }
        if (op == atomic_op::cas) { atomicCAS(bits, 0u, __float_as_uint(1.0f)); return; }
        auto old = atomicCAS(bits, 0u, 0u);
        unsigned int assumed;
        do {
            assumed = old;
            const float next = op == atomic_op::minimum ?
                fminf(__uint_as_float(old), value) : fmaxf(__uint_as_float(old), value);
            old = atomicCAS(bits, assumed, __float_as_uint(next));
        } while (old != assumed);
    } else {
        auto* bits = reinterpret_cast<unsigned long long*>(address);
        if (op == atomic_op::exchange) {
            atomicExch(bits, static_cast<unsigned long long>(__double_as_longlong(7.0)));
            return;
        }
        if (op == atomic_op::cas) {
            atomicCAS(bits, 0ull, static_cast<unsigned long long>(__double_as_longlong(1.0)));
            return;
        }
        auto old = atomicCAS(bits, 0ull, 0ull);
        unsigned long long assumed;
        do {
            assumed = old;
            const double current = __longlong_as_double(static_cast<long long>(old));
            const double next = op == atomic_op::minimum ? fmin(current, value) : fmax(current, value);
            old = atomicCAS(bits, assumed, static_cast<unsigned long long>(__double_as_longlong(next)));
        } while (old != assumed);
    }
}

template <class T>
__global__ void atomic_kernel(T* values, std::size_t elements, std::size_t addresses,
                              unsigned iterations, atomic_op op) {
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x; i < elements; i += step) {
        for (unsigned k = 0; k < iterations; ++k) {
            const auto incoming = op == atomic_op::add ? 1u : static_cast<unsigned>(i % 31) + 1u;
            if constexpr (std::is_integral<T>::value)
                integer_atomic(values + i % addresses, static_cast<T>(incoming), op);
            else if constexpr (std::is_same<T, __half>::value)
                atomicAdd(values + i % addresses, __float2half(1.0f));
            else if constexpr (std::is_same<T, __nv_bfloat16>::value)
                atomicAdd(values + i % addresses, __float2bfloat16(1.0f));
            else
                float_atomic(values + i % addresses, static_cast<T>(incoming), op);
        }
    }
}

template <class T> T make_atomic_value(long double x) { return static_cast<T>(x); }
template <> __half make_atomic_value<__half>(long double x) { return __float2half(static_cast<float>(x)); }
template <> __nv_bfloat16 make_atomic_value<__nv_bfloat16>(long double x) {
    return __float2bfloat16(static_cast<float>(x));
}
template <class T> long double read_atomic_value(T x) { return static_cast<long double>(x); }
template <> long double read_atomic_value(__half x) { return __half2float(x); }
template <> long double read_atomic_value(__nv_bfloat16 x) { return __bfloat162float(x); }

template <class T>
void atomic_case(context& ctx, result& r, atomic_op op, std::size_t n,
                  std::size_t addresses, unsigned iterations) {
    const auto padding = sizeof(T) < 4 ? 4 / sizeof(T) : 1;
    const auto storage_count = (addresses + padding - 1) / padding * padding;
    ctx.memory_guard(checked_product(storage_count, sizeof(T)));
    device_buffer<T> values(storage_count);
    long double initial = 0;
    if (op == atomic_op::minimum) {
        if constexpr (std::is_integral<T>::value)
            initial = std::min<long double>(std::numeric_limits<T>::max(), 1000);
        else initial = 1000;
    }
    if (op == atomic_op::maximum) initial = 0;
    std::vector<T> init(storage_count, make_atomic_value<T>(initial));
    if constexpr (std::is_integral<T>::value) {
        if (op == atomic_op::bit_and)
            std::fill(init.begin(), init.end(), static_cast<T>(~T(0)));
    }
    upload(ctx, values.get(), init);
    const int threads = thread_count(ctx);
    const int blocks = grid_count(ctx, n, threads);
    auto launch = [&] {
        atomic_kernel<<<blocks, threads, 0, ctx.stream>>>(
            values.get(), n, addresses, iterations, op);
        check_launch();
    };
    r.samples_ms = ctx.measure(launch);
    // Independent fresh-state validation: timed launch repetition is not a reference.
    upload(ctx, values.get(), init);
    launch();
    const auto got = download(ctx, values.get(), addresses);
    begin_validation(r, addresses);
    for (std::size_t a = 0; a < addresses; ++a) {
        const auto participants = (n - 1 - a) / addresses + 1;
        const auto updates = static_cast<unsigned long long>(participants) * iterations;
        long double ref = initial;
        if (op == atomic_op::add) {
            if constexpr (std::is_same<T, __half>::value)
                ref = std::min<unsigned long long>(updates, 2048);
            else if constexpr (std::is_same<T, __nv_bfloat16>::value)
                ref = std::min<unsigned long long>(updates, 256);
            else if constexpr (std::is_same<T, float>::value)
                ref = std::min<unsigned long long>(updates, 16777216);
            else if constexpr (std::is_integral<T>::value) {
                using unsigned_type = typename std::make_unsigned<T>::type;
                const auto bits = static_cast<unsigned_type>(updates);
                T signed_or_unsigned{};
                std::memcpy(&signed_or_unsigned, &bits, sizeof(T));
                ref = static_cast<long double>(signed_or_unsigned);
            } else ref = static_cast<long double>(updates);
        } else if (op == atomic_op::minimum || op == atomic_op::maximum) {
            for (std::size_t i = a; i < n; i += addresses) {
                const auto v = static_cast<long double>(i % 31 + 1);
                ref = op == atomic_op::minimum ? std::min(ref, v) : std::max(ref, v);
            }
        } else if (op == atomic_op::cas) ref = 1;
        else if (op == atomic_op::exchange) ref = 7;
        else if (op == atomic_op::bit_xor) ref = updates % 2 ? 0x55 : 0;
        else ref = 0x55;
        check_value(r, read_atomic_value(got[a]), ref);
    }
    r.work_per_iteration = static_cast<double>(n) * iterations;
    // Actual physical memory traffic depends on contention and caching.
    r.bytes_per_iteration = 0;
    r.unit = "Gatomic/s";
    r.metadata["work_domain"] = "atomic";
    r.metrics["atomic_addresses"] = static_cast<double>(addresses);
    r.metrics["updates_per_address_mean"] = static_cast<double>(n) * iterations / addresses;
    r.metadata["atomic_scope"] = "device/global relaxed legacy atomic API";
    r.metadata["timed_state"] = "updates persist between repetitions; validation uses a separate fresh-state launch";
    if (op == atomic_op::cas)
        r.metadata["cas_semantics"] = "compare 0, exchange 1; steady-state failed comparisons";
    if constexpr (std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value)
        r.metadata["low_precision_add"] = "all +1; saturation of unit increments is expected and checked";
    if constexpr (std::is_integral<T>::value && sizeof(T) < 4)
        r.metadata["packed_atomic"] = "emulated with aligned 32-bit CAS preserving adjacent subword fields";
}

bool run_atomic(context& ctx, result& r) {
    const auto op_name = ctx.opts.get("op", "add");
    atomic_op op;
    if (op_name == "add") op = atomic_op::add;
    else if (op_name == "min") op = atomic_op::minimum;
    else if (op_name == "max") op = atomic_op::maximum;
    else if (op_name == "exchange") op = atomic_op::exchange;
    else if (op_name == "cas") op = atomic_op::cas;
    else if (op_name == "and") op = atomic_op::bit_and;
    else if (op_name == "or") op = atomic_op::bit_or;
    else if (op_name == "xor") op = atomic_op::bit_xor;
    else { skip(r, "unknown atomic operation: " + op_name); return true; }
    const auto dtype = ctx.opts.get("dtype", "u32");
    const auto n = positive_size(ctx.opts, "elements", 1 << 18);
    const auto addresses = positive_size(ctx.opts, "addresses", n);
    const auto iterations = positive_size(ctx.opts, "iterations", 4);
    if (addresses > n || iterations > (1u << 20) ||
        n > std::numeric_limits<unsigned long long>::max() / iterations)
        throw std::invalid_argument("atomics require addresses <= elements and iterations <=2^20 without count overflow");
    const bool bitwise = op == atomic_op::bit_and || op == atomic_op::bit_or || op == atomic_op::bit_xor;
    const bool floating = dtype == "f16" || dtype == "bf16" || dtype == "f32" || dtype == "f64";
    if ((floating && bitwise) || ((dtype == "f16" || dtype == "bf16") && op != atomic_op::add)) {
        skip(r, "selected atomic operation is not implemented for this floating-point type");
        return true;
    }
    r.execution = (floating && op != atomic_op::add) || dtype == "i8" || dtype == "u8" ||
                  dtype == "i16" || dtype == "u16" ? "emulated" : "compiler_lowered";
    if (dtype == "i8") atomic_case<signed char>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "u8") atomic_case<unsigned char>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "i16") atomic_case<short>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "u16") atomic_case<unsigned short>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "i32") atomic_case<int>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "u32") atomic_case<unsigned>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "i64") atomic_case<long long>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "u64") atomic_case<unsigned long long>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "f32") atomic_case<float>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "f64") atomic_case<double>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "f16") atomic_case<__half>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else if (dtype == "bf16") atomic_case<__nv_bfloat16>(ctx, r, op, n, addresses, static_cast<unsigned>(iterations));
    else skip(r, "unknown atomic dtype: " + dtype);
    return true;
}

__global__ void warp_shuffle_kernel(const float* input, float* output,
                                    std::size_t n, unsigned iterations) {
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockDim.x;
         base < n; base += step) {
        const auto i = base + threadIdx.x;
        float value = i < n ? input[i] : 0.0f;
        for (unsigned k = 0; k < iterations; ++k)
            value = __shfl_sync(0xffffffffu, value, (threadIdx.x + 1) & 31);
        if (i < n) output[i] = value;
    }
}

__global__ void warp_ballot_kernel(const float* input, unsigned* output, std::size_t n) {
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t base = static_cast<std::size_t>(blockIdx.x) * blockDim.x;
         base < n; base += step) {
        const auto i = base + threadIdx.x;
        const bool positive = i < n && input[i] > 0;
        const unsigned ballot = __ballot_sync(0xffffffffu, positive);
        if (i < n) output[i] = ballot;
    }
}

template <int threads, bool scan>
__global__ void block_collective_kernel(const float* input, float* output, std::size_t n) {
    using reducer = cub::BlockReduce<float, threads>;
    using scanner = cub::BlockScan<float, threads>;
    __shared__ typename reducer::TempStorage reduction_storage;
    __shared__ typename scanner::TempStorage scan_storage;
    const std::size_t block_start = static_cast<std::size_t>(blockIdx.x) * threads;
    const auto index = block_start + threadIdx.x;
    const float value = index < n ? input[index] : 0.0f;
    if constexpr (scan) {
        float prefix;
        scanner(scan_storage).InclusiveSum(value, prefix);
        if (index < n) output[index] = prefix;
    } else {
        const float sum = reducer(reduction_storage).Sum(value);
        if (threadIdx.x == 0) output[blockIdx.x] = sum;
    }
}

template <bool scan>
void launch_block_collective(context& ctx, const float* input, float* output,
                             std::size_t n, int threads, int blocks) {
    switch (threads) {
    case 32: block_collective_kernel<32, scan><<<blocks, 32, 0, ctx.stream>>>(input, output, n); break;
    case 64: block_collective_kernel<64, scan><<<blocks, 64, 0, ctx.stream>>>(input, output, n); break;
    case 128: block_collective_kernel<128, scan><<<blocks, 128, 0, ctx.stream>>>(input, output, n); break;
    case 256: block_collective_kernel<256, scan><<<blocks, 256, 0, ctx.stream>>>(input, output, n); break;
    case 512: block_collective_kernel<512, scan><<<blocks, 512, 0, ctx.stream>>>(input, output, n); break;
    case 1024: block_collective_kernel<1024, scan><<<blocks, 1024, 0, ctx.stream>>>(input, output, n); break;
    default: throw std::invalid_argument("unsupported block-collective thread count");
    }
    check_launch();
}

__global__ void barrier_kernel(unsigned* output, unsigned iterations) {
    __shared__ unsigned values[1024];
    values[threadIdx.x] = threadIdx.x;
    __syncthreads();
    unsigned total = 0;
    for (unsigned k = 0; k < iterations; ++k) {
        total += values[(threadIdx.x + 1) % blockDim.x];
        __syncthreads();
        values[threadIdx.x] += 1;
        __syncthreads();
    }
    output[static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x] = total;
}

bool run_collective(context& ctx, result& r) {
    const auto op = ctx.opts.get("op", "reduce_sum");
    const auto dtype = ctx.opts.get("dtype", "f32");
    if (dtype != "f32" && !(dtype == "u32" && (op == "ballot" || op == "barrier"))) {
        skip(r, "collectives use f32 data; ballot/barrier may be labelled u32 output");
        return true;
    }
    const auto n = positive_size(ctx.opts, "elements", 1 << 20);
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("CUB baseline uses a 32-bit item count");
    const int threads = thread_count(ctx, true);
    const int blocks = grid_count(ctx, n, threads);
    const auto data = float_input(n);
    ctx.memory_guard(checked_product(n, 4 * sizeof(float)));
    device_buffer<float> input(n), output(n);
    upload(ctx, input.get(), data);
    r.execution = "compiler_lowered";
    r.unit = "Gitem/s";
    r.metadata["work_domain"] = "item";
    r.work_per_iteration = static_cast<double>(n);
    r.bytes_per_iteration = static_cast<double>(2 * n * sizeof(float));
    if (op == "shuffle") {
        const auto iterations = positive_size(ctx.opts, "iterations", 32);
        if (iterations > (1u << 20)) throw std::invalid_argument("shuffle iterations >2^20");
        auto launch = [&] {
            warp_shuffle_kernel<<<blocks, threads, 0, ctx.stream>>>(
                input.get(), output.get(), n, static_cast<unsigned>(iterations));
            check_launch();
        };
        r.samples_ms = ctx.measure(launch);
        const auto got = download(ctx, output.get(), n);
        begin_validation(r, n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto source = (i / 32) * 32 + (i + iterations) % 32;
            check_value(r, got[i], source < n ? data[source] : 0);
        }
        r.work_per_iteration *= iterations;
        r.metadata["primitive"] = "dependent warp lane rotation; padded tail lanes contribute zero";
        return true;
    }
    if (op == "ballot") {
        device_buffer<unsigned> bits(n);
        auto launch = [&] {
            warp_ballot_kernel<<<blocks, threads, 0, ctx.stream>>>(input.get(), bits.get(), n);
            check_launch();
        };
        r.samples_ms = ctx.measure(launch);
        const auto got = download(ctx, bits.get(), n);
        begin_validation(r, n);
        for (std::size_t base = 0; base < n; base += 32) {
            unsigned ref = 0;
            for (unsigned lane = 0; lane < 32 && base + lane < n; ++lane)
                if (data[base + lane] > 0) ref |= 1u << lane;
            for (unsigned lane = 0; lane < 32 && base + lane < n; ++lane)
                check_value(r, got[base + lane], ref);
        }
        return true;
    }
    if (op == "barrier") {
        const auto iterations = positive_size(ctx.opts, "iterations", 1024);
        if (iterations > (1u << 20)) throw std::invalid_argument("barrier iterations >2^20");
        const auto count = static_cast<std::size_t>(blocks) * threads;
        device_buffer<unsigned> values(count);
        auto launch = [&] {
            barrier_kernel<<<blocks, threads, 0, ctx.stream>>>(values.get(), static_cast<unsigned>(iterations));
            check_launch();
        };
        r.samples_ms = ctx.measure(launch);
        const auto got = download(ctx, values.get(), count);
        begin_validation(r, count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto neighbor = (i % threads + 1) % threads;
            const auto ref = static_cast<unsigned>(iterations * neighbor + iterations * (iterations - 1) / 2);
            check_value(r, got[i], ref);
        }
        r.work_per_iteration = static_cast<double>(blocks) * (2 * iterations + 1);
        r.bytes_per_iteration = 0;
        r.unit = "Gbarrier/s";
        r.metadata["work_domain"] = "barrier";
        r.metadata["operation_count"] = "CTA-wide barrier instances; includes dependent shared loads/stores";
        return true;
    }
    if (op == "block_reduce" || op == "block_scan") {
        const auto all_blocks = static_cast<int>((n + threads - 1) / threads);
        auto launch = [&] {
            if (op == "block_scan")
                launch_block_collective<true>(ctx, input.get(), output.get(), n, threads, all_blocks);
            else
                launch_block_collective<false>(ctx, input.get(), output.get(), n, threads, all_blocks);
        };
        r.samples_ms = ctx.measure(launch);
        const auto output_count = op == "block_scan" ? n : static_cast<std::size_t>(all_blocks);
        const auto got = download(ctx, output.get(), output_count);
        begin_validation(r, output_count);
        for (int block = 0; block < all_blocks; ++block) {
            long double sum = 0;
            for (int t = 0; t < threads; ++t) {
                const auto i = static_cast<std::size_t>(block) * threads + t;
                if (i >= n) break;
                sum += data[i];
                if (op == "block_scan") check_value(r, got[i], sum, 1.0e-4L, 1.0e-6L);
            }
            if (op == "block_reduce") check_value(r, got[block], sum, 1.0e-4L, 1.0e-6L);
        }
        r.execution = "library_selected";
        r.metadata["library"] = "CUB block primitive";
        r.metadata["scope"] = "each block independently; block_scan is not a device-wide scan";
        return true;
    }
    if (op == "reduce_sum" || op == "scan" || op == "sort") {
        std::size_t temp_bytes = 0;
        const int count = static_cast<int>(n);
        if (op == "reduce_sum")
            CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_bytes, input.get(), output.get(), count, ctx.stream));
        else if (op == "scan")
            CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, input.get(), output.get(), count, ctx.stream));
        else
            CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, input.get(), output.get(), count, 0, 32, ctx.stream));
        ctx.memory_guard(n * sizeof(float) * 2 + temp_bytes);
        device_buffer<unsigned char> temp(std::max<std::size_t>(temp_bytes, 1));
        auto launch = [&] {
            if (op == "reduce_sum")
                CUDA_CHECK(cub::DeviceReduce::Sum(temp.get(), temp_bytes, input.get(), output.get(), count, ctx.stream));
            else if (op == "scan")
                CUDA_CHECK(cub::DeviceScan::InclusiveSum(temp.get(), temp_bytes, input.get(), output.get(), count, ctx.stream));
            else
                CUDA_CHECK(cub::DeviceRadixSort::SortKeys(temp.get(), temp_bytes, input.get(), output.get(), count, 0, 32, ctx.stream));
        };
        r.samples_ms = ctx.measure(launch);
        const auto output_count = op == "reduce_sum" ? 1 : n;
        const auto got = download(ctx, output.get(), output_count);
        begin_validation(r, output_count);
        if (op == "sort") {
            auto ref = data;
            std::sort(ref.begin(), ref.end());
            for (std::size_t i = 0; i < n; ++i) check_value(r, got[i], ref[i]);
        } else {
            long double sum = 0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += data[i];
                if (op == "scan") check_value(r, got[i], sum, 1.0e-3L, 1.0e-5L);
            }
            if (op == "reduce_sum") check_value(r, got[0], sum, 1.0e-3L, 1.0e-5L);
        }
        r.execution = "library_selected";
        r.metadata["library"] = "CUB device primitive";
        r.metrics["temporary_storage_bytes"] = static_cast<double>(temp_bytes);
        r.metadata["bandwidth_scope"] = "minimum logical input/output bytes; CUB internal passes are excluded";
        r.bytes_per_iteration = static_cast<double>((n + output_count) * sizeof(float));
        return true;
    }
    skip(r, "unknown collective operation: " + op);
    return true;
}

template <class T> struct storage_math { using accumulator = float; };
template <> struct storage_math<double> { using accumulator = double; };

template <class T>
__host__ __device__ typename storage_math<T>::accumulator load_value(T x) {
    return static_cast<typename storage_math<T>::accumulator>(x);
}
template <>
__host__ __device__ float load_value(__half x) { return __half2float(x); }
template <>
__host__ __device__ float load_value(__nv_bfloat16 x) { return __bfloat162float(x); }

template <class T, class A>
__host__ __device__ T store_value(A x) { return static_cast<T>(x); }
template <>
__host__ __device__ __half store_value<__half, float>(float x) { return __float2half_rn(x); }
template <>
__host__ __device__ __nv_bfloat16 store_value<__nv_bfloat16, float>(float x) {
    return __float2bfloat16_rn(x);
}

template <class T>
std::vector<T> typed_input(std::size_t n, unsigned salt = 0, bool positive = false) {
    using accumulator = typename storage_math<T>::accumulator;
    std::vector<T> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto v = static_cast<accumulator>(input_value(i, salt));
        if (positive) v = v * accumulator(0.125) + accumulator(1);
        x[i] = store_value<T>(v);
    }
    return x;
}

template <class A>
__device__ A shared_sum(A value, A* temporary) {
    temporary[threadIdx.x] = value;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset; offset /= 2) {
        if (threadIdx.x < offset) temporary[threadIdx.x] += temporary[threadIdx.x + offset];
        __syncthreads();
    }
    const A total = temporary[0];
    __syncthreads();
    return total;
}

template <class A>
__device__ A shared_max(A value, A* temporary) {
    temporary[threadIdx.x] = value;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset; offset /= 2) {
        if (threadIdx.x < offset) {
            const A next = temporary[threadIdx.x + offset];
            if (next > temporary[threadIdx.x]) temporary[threadIdx.x] = next;
        }
        __syncthreads();
    }
    const A maximum = temporary[0];
    __syncthreads();
    return maximum;
}

enum class row_op { softmax, rmsnorm, layernorm };

template <class T>
__global__ void row_workload_kernel(const T* input, const T* gamma, const T* beta,
                                   T* output, std::size_t cols, row_op op) {
    using A = typename storage_math<T>::accumulator;
    extern __shared__ __align__(16) unsigned char temporary_raw[];
    auto* temporary = reinterpret_cast<A*>(temporary_raw);
    const auto row_start = static_cast<std::size_t>(blockIdx.x) * cols;
    if (op == row_op::softmax) {
        A maximum = -static_cast<A>(INFINITY);
        for (std::size_t j = threadIdx.x; j < cols; j += blockDim.x)
            maximum = fmax(maximum, load_value(input[row_start + j]));
        maximum = shared_max(maximum, temporary);
        A total = 0;
        for (std::size_t j = threadIdx.x; j < cols; j += blockDim.x)
            total += exp(load_value(input[row_start + j]) - maximum);
        total = shared_sum(total, temporary);
        for (std::size_t j = threadIdx.x; j < cols; j += blockDim.x)
            output[row_start + j] = store_value<T>(
                exp(load_value(input[row_start + j]) - maximum) / total);
    } else {
        A mean = 0;
        if (op == row_op::layernorm) {
            A sum = 0;
            for (std::size_t j = threadIdx.x; j < cols; j += blockDim.x)
                sum += load_value(input[row_start + j]);
            mean = shared_sum(sum, temporary) / static_cast<A>(cols);
        }
        A squares = 0;
        for (std::size_t j = threadIdx.x; j < cols; j += blockDim.x) {
            const A difference = load_value(input[row_start + j]) - mean;
            squares += difference * difference;
        }
        const A variance = shared_sum(squares, temporary) / static_cast<A>(cols);
        const A inverse = A(1) / sqrt(variance + A(1.0e-5));
        for (std::size_t j = threadIdx.x; j < cols; j += blockDim.x) {
            A value = (load_value(input[row_start + j]) - mean) * inverse * load_value(gamma[j]);
            if (op == row_op::layernorm) value += load_value(beta[j]);
            output[row_start + j] = store_value<T>(value);
        }
    }
}

enum class activation_op { silu, gelu, gelu_tanh, swiglu, relu };

template <class T>
__global__ void activation_kernel(const T* x, const T* z, T* y, std::size_t n,
                                  activation_op op) {
    using A = typename storage_math<T>::accumulator;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += step) {
        const A value = load_value(x[i]);
        A out;
        switch (op) {
        case activation_op::relu: out = fmax(value, A(0)); break;
        case activation_op::silu: out = value / (A(1) + exp(-value)); break;
        case activation_op::swiglu: out = value / (A(1) + exp(-value)) * load_value(z[i]); break;
        case activation_op::gelu: out = A(0.5) * value * (A(1) + erf(value * A(0.7071067811865475244))); break;
        case activation_op::gelu_tanh:
            out = A(0.5) * value * (A(1) + tanh(A(0.7978845608028653559) *
                   (value + A(0.044715) * value * value * value))); break;
        }
        y[i] = store_value<T>(out);
    }
}

template <class T>
__global__ void rope_kernel(const T* input, T* output, std::size_t n, std::size_t cols) {
    using A = typename storage_math<T>::accumulator;
    const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < n / 2; pair += step) {
        const auto index = pair * 2;
        const auto position = index / cols;
        const auto dimension = index % cols;
        const A angle = static_cast<A>(position) * pow(A(10000), -static_cast<A>(dimension) / static_cast<A>(cols));
        const A cosine = cos(angle), sine = sin(angle);
        const A a = load_value(input[index]), b = load_value(input[index + 1]);
        output[index] = store_value<T>(a * cosine - b * sine);
        output[index + 1] = store_value<T>(a * sine + b * cosine);
    }
}

template <class T>
std::pair<long double, long double> workload_tolerance() {
    if constexpr (std::is_same<T, __half>::value) return {2.0e-3L, 3.0e-3L};
    if constexpr (std::is_same<T, __nv_bfloat16>::value) return {1.2e-2L, 1.2e-2L};
    if constexpr (std::is_same<T, double>::value) return {2.0e-11L, 2.0e-11L};
    return {2.0e-5L, 5.0e-5L};
}

template <class T>
void simple_workload_case(context& ctx, result& r, const std::string& op) {
    using A = typename storage_math<T>::accumulator;
    const auto rows = positive_size(ctx.opts, "rows", 128);
    const auto cols = positive_size(ctx.opts, "cols", 1024);
    const bool row = op == "softmax" || op == "rmsnorm" || op == "layernorm";
    const auto product = checked_product(rows, cols);
    const auto n = (row || op == "rope") ? product :
        positive_size(ctx.opts, "elements", static_cast<std::int64_t>(product));
    if (rows > static_cast<std::size_t>(ctx.prop.maxGridSize[0]))
        throw std::invalid_argument("row count exceeds device grid limit");
    if (op == "rope" && cols % 2) {
        skip(r, "interleaved-pair RoPE requires even cols");
        return;
    }
    ctx.memory_guard(checked_product(n, 3 * sizeof(T)) + checked_product(cols, 2 * sizeof(T)));
    const auto input = typed_input<T>(n, 0), other = typed_input<T>(n, 5);
    const auto gamma = typed_input<T>(cols, 7, true), beta = typed_input<T>(cols, 9);
    device_buffer<T> dx(n), dz(n), dy(n), dg(cols), db(cols);
    upload(ctx, dx.get(), input); upload(ctx, dz.get(), other);
    upload(ctx, dg.get(), gamma); upload(ctx, db.get(), beta);
    const int threads = thread_count(ctx, true);
    const int blocks = grid_count(ctx, n, threads);
    row_op row_kind = op == "softmax" ? row_op::softmax :
                      (op == "rmsnorm" ? row_op::rmsnorm : row_op::layernorm);
    activation_op activation = activation_op::silu;
    if (op == "gelu") activation = activation_op::gelu;
    else if (op == "gelu_tanh") activation = activation_op::gelu_tanh;
    else if (op == "swiglu") activation = activation_op::swiglu;
    else if (op == "relu") activation = activation_op::relu;
    auto launch = [&] {
        if (row)
            row_workload_kernel<<<static_cast<int>(rows), threads, threads * sizeof(A), ctx.stream>>>(
                dx.get(), dg.get(), db.get(), dy.get(), cols, row_kind);
        else if (op == "rope")
            rope_kernel<<<blocks, threads, 0, ctx.stream>>>(dx.get(), dy.get(), n, cols);
        else
            activation_kernel<<<blocks, threads, 0, ctx.stream>>>(dx.get(), dz.get(), dy.get(), n, activation);
        check_launch();
    };
    r.samples_ms = ctx.measure(launch);
    const auto got = download(ctx, dy.get(), n);
    const auto tolerance = workload_tolerance<T>();
    r.metrics["validation_atol"] = static_cast<double>(tolerance.first);
    r.metrics["validation_rtol"] = static_cast<double>(tolerance.second);
    begin_validation(r, n);
    if (row) {
        for (std::size_t i = 0; i < rows; ++i) {
            long double mean = 0, maximum = -std::numeric_limits<long double>::infinity();
            for (std::size_t j = 0; j < cols; ++j) {
                const auto value = static_cast<long double>(load_value(input[i * cols + j]));
                mean += value; maximum = std::max(maximum, value);
            }
            mean = op == "layernorm" ? mean / cols : 0;
            long double sum = 0;
            for (std::size_t j = 0; j < cols; ++j) {
                const auto value = static_cast<long double>(load_value(input[i * cols + j]));
                sum += op == "softmax" ? std::exp(value - maximum) : (value - mean) * (value - mean);
            }
            const auto inverse = 1.0L / std::sqrt(sum / cols + 1.0e-5L);
            for (std::size_t j = 0; j < cols; ++j) {
                const auto value = static_cast<long double>(load_value(input[i * cols + j]));
                long double expected = op == "softmax" ? std::exp(value - maximum) / sum :
                    (value - mean) * inverse * load_value(gamma[j]);
                if (op == "layernorm") expected += load_value(beta[j]);
                check_value(r, load_value(got[i * cols + j]), expected, tolerance.first, tolerance.second);
            }
        }
    } else if (op == "rope") {
        for (std::size_t i = 0; i < n; i += 2) {
            const auto angle = static_cast<long double>(i / cols) *
                std::pow(10000.0L, -static_cast<long double>(i % cols) / cols);
            const auto cosine = std::cos(angle), sine = std::sin(angle);
            const auto a = static_cast<long double>(load_value(input[i]));
            const auto b = static_cast<long double>(load_value(input[i + 1]));
            check_value(r, load_value(got[i]), a * cosine - b * sine, tolerance.first, tolerance.second);
            check_value(r, load_value(got[i + 1]), a * sine + b * cosine, tolerance.first, tolerance.second);
        }
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            const auto x = static_cast<long double>(load_value(input[i]));
            long double expected;
            if (op == "relu") expected = std::max(x, 0.0L);
            else if (op == "gelu") expected = 0.5L * x * (1 + std::erf(x / std::sqrt(2.0L)));
            else if (op == "gelu_tanh") expected = 0.5L * x * (1 + std::tanh(
                std::sqrt(2.0L / std::acos(-1.0L)) * (x + 0.044715L * x * x * x)));
            else {
                expected = x / (1 + std::exp(-x));
                if (op == "swiglu") expected *= load_value(other[i]);
            }
            check_value(r, load_value(got[i]), expected, tolerance.first, tolerance.second);
        }
    }
    r.execution = "compiler_lowered";
    r.unit = "Gitem/s";
    r.metadata["work_domain"] = "item";
    r.work_per_iteration = static_cast<double>(n);
    r.bytes_per_iteration = static_cast<double>((op == "swiglu" ? 3 : 2) * n * sizeof(T));
    r.metadata["implementation"] = "portable CUDA baseline; not a vendor fused kernel";
    r.metadata["accumulator"] = std::is_same<T, double>::value ? "f64" : "f32";
    r.metadata["bandwidth_scope"] = "minimum logical tensor traffic; repeated passes and affine loads excluded";
    if (row) r.metrics["epsilon"] = 1.0e-5;
    if (op == "rope") r.metadata["rope_convention"] = "interleaved pairs; row index is position; base 10000; trig evaluated in timed kernel";
}

template <class T>
__global__ void attention_score_kernel(const T* q, const T* k,
                                       typename storage_math<T>::accumulator* scores,
                                       std::size_t total_scores, std::size_t sequence,
                                       std::size_t dimension, bool causal) {
    using A = typename storage_math<T>::accumulator;
    const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total_scores; i += step) {
        const auto group = i / (sequence * sequence);
        const auto query = (i / sequence) % sequence;
        const auto key = i % sequence;
        if (causal && key > query) { scores[i] = -static_cast<A>(INFINITY); continue; }
        A sum = 0;
        for (std::size_t d = 0; d < dimension; ++d)
            sum += load_value(q[(group * sequence + query) * dimension + d]) *
                   load_value(k[(group * sequence + key) * dimension + d]);
        scores[i] = sum / sqrt(static_cast<A>(dimension));
    }
}

template <class T>
__global__ void attention_output_kernel(const typename storage_math<T>::accumulator* probabilities,
                                        const T* values, T* output, std::size_t elements,
                                        std::size_t sequence, std::size_t dimension) {
    using A = typename storage_math<T>::accumulator;
    const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += step) {
        const auto row = i / dimension;
        const auto group = row / sequence;
        const auto d = i % dimension;
        A sum = 0;
        for (std::size_t j = 0; j < sequence; ++j)
            sum += probabilities[row * sequence + j] *
                   load_value(values[(group * sequence + j) * dimension + d]);
        output[i] = store_value<T>(sum);
    }
}

template <class T>
void attention_case(context& ctx, result& r) {
    using A = typename storage_math<T>::accumulator;
    const auto sequence = positive_size(ctx.opts, "sequence", 128);
    const auto dimension = positive_size(ctx.opts, "head-dim", 64);
    const auto batch = positive_size(ctx.opts, "batch", 1);
    const auto heads = positive_size(ctx.opts, "heads", 4);
    const bool causal = ctx.opts.flag("causal", false);
    const auto groups = checked_product(batch, heads);
    const auto rows = checked_product(groups, sequence);
    const auto elements = checked_product(rows, dimension);
    const auto score_count = checked_product(rows, sequence);
    if (rows > static_cast<std::size_t>(ctx.prop.maxGridSize[0]))
        throw std::invalid_argument("attention row count exceeds grid limit");
    ctx.memory_guard(checked_product(elements, 4 * sizeof(T)) + checked_product(score_count, 2 * sizeof(A)));
    const auto q = typed_input<T>(elements, 1), k = typed_input<T>(elements, 4), v = typed_input<T>(elements, 17);
    device_buffer<T> dq(elements), dk(elements), dv(elements), output(elements);
    device_buffer<A> scores(score_count), probabilities(score_count);
    upload(ctx, dq.get(), q); upload(ctx, dk.get(), k); upload(ctx, dv.get(), v);
    const int threads = thread_count(ctx, true);
    const int blocks = grid_count(ctx, score_count, threads);
    auto launch = [&] {
        attention_score_kernel<<<blocks, threads, 0, ctx.stream>>>(
            dq.get(), dk.get(), scores.get(), score_count, sequence, dimension, causal);
        check_launch();
        row_workload_kernel<<<static_cast<int>(rows), threads, threads * sizeof(A), ctx.stream>>>(
            scores.get(), static_cast<const A*>(nullptr), static_cast<const A*>(nullptr),
            probabilities.get(), sequence, row_op::softmax);
        check_launch();
        attention_output_kernel<<<blocks, threads, 0, ctx.stream>>>(
            probabilities.get(), dv.get(), output.get(), elements, sequence, dimension);
        check_launch();
    };
    r.samples_ms = ctx.measure(launch);
    const auto got = download(ctx, output.get(), elements);
    const auto row_checks = std::min(rows, positive_size(ctx.opts, "check-rows", 16));
    begin_validation(r, row_checks * dimension);
    r.metadata["validation_scope"] = row_checks == rows ? "all_output_elements" : "evenly_spaced_rows_all_channels";
    const auto tolerance = workload_tolerance<T>();
    r.metrics["validation_atol"] = static_cast<double>(tolerance.first * 4);
    r.metrics["validation_rtol"] = static_cast<double>(tolerance.second * 4);
    std::vector<long double> weights(sequence);
    for (std::size_t sample = 0; sample < row_checks; ++sample) {
        const auto row = row_checks == 1 ? 0 : sample * (rows - 1) / (row_checks - 1);
        const auto group = row / sequence, query = row % sequence;
        long double maximum = -std::numeric_limits<long double>::infinity();
        for (std::size_t j = 0; j < sequence; ++j) {
            if (causal && j > query) { weights[j] = -std::numeric_limits<long double>::infinity(); continue; }
            long double dot = 0;
            for (std::size_t d = 0; d < dimension; ++d)
                dot += static_cast<long double>(load_value(q[row * dimension + d])) *
                       load_value(k[(group * sequence + j) * dimension + d]);
            weights[j] = dot / std::sqrt(static_cast<long double>(dimension));
            maximum = std::max(maximum, weights[j]);
        }
        long double total = 0;
        for (auto& weight : weights) { weight = std::exp(weight - maximum); total += weight; }
        for (std::size_t d = 0; d < dimension; ++d) {
            long double expected = 0;
            for (std::size_t j = 0; j < sequence; ++j)
                expected += weights[j] / total * load_value(v[(group * sequence + j) * dimension + d]);
            check_value(r, load_value(got[row * dimension + d]), expected,
                        tolerance.first * 4, tolerance.second * 4);
        }
    }
    r.execution = "compiler_lowered";
    r.unit = "Gflop/s";
    r.metadata["work_domain"] = "floating_point";
    // QK skips masked work, AV still visits all keys. Report exactly those loop FMAs.
    const double qk_pairs = static_cast<double>(groups) * sequence *
                           (causal ? (static_cast<double>(sequence) + 1) / 2 : sequence);
    r.work_per_iteration = 2.0 * dimension * (qk_pairs + static_cast<double>(score_count));
    r.bytes_per_iteration = static_cast<double>(4 * elements * sizeof(T) + 4 * score_count * sizeof(A));
    r.metadata["implementation"] = "three-stage naive QK/softmax/AV CUDA baseline; not FlashAttention";
    r.metadata["operation_count"] = "QK and AV multiply-adds only; softmax/SFU and indexing work excluded";
    r.metadata["bandwidth_scope"] = "logical tensor traffic lower bound; naive rereads excluded";
    r.metadata["accumulator"] = std::is_same<T, double>::value ? "f64" : "f32";
}

bool run_workload(context& ctx, result& r) {
    const auto op = ctx.opts.get("op", "softmax");
    const auto dtype = ctx.opts.get("dtype", "f32");
    const bool recognized = op == "softmax" || op == "rmsnorm" || op == "layernorm" ||
        op == "rope" || op == "gelu" || op == "gelu_tanh" || op == "silu" ||
        op == "swiglu" || op == "relu" || op == "attention";
    if (!recognized) { skip(r, "unknown workload operation: " + op); return true; }
    if (dtype == "f16") {
        if (op == "attention") attention_case<__half>(ctx, r);
        else simple_workload_case<__half>(ctx, r, op);
    } else if (dtype == "bf16") {
        if (op == "attention") attention_case<__nv_bfloat16>(ctx, r);
        else simple_workload_case<__nv_bfloat16>(ctx, r, op);
    } else if (dtype == "f32") {
        if (op == "attention") attention_case<float>(ctx, r);
        else simple_workload_case<float>(ctx, r, op);
    } else if (dtype == "f64") {
        if (op == "attention") attention_case<double>(ctx, r);
        else simple_workload_case<double>(ctx, r, op);
    } else skip(r, "portable workloads support f16, bf16, f32, f64 storage");
    return true;
}

__global__ void empty_kernel() { asm volatile(""); }

struct graph_owner {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~graph_owner() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};

double median_value(std::vector<double> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5;
}

bool run_launch(context& ctx, result& r) {
    const auto op = ctx.opts.get("op", "empty");
    if (op != "empty" && op != "graph") {
        skip(r, "launch operations are empty and graph"); return true;
    }
    const auto iterations = positive_size(ctx.opts, "iterations", 100);
    const auto nodes = positive_size(ctx.opts, "graph-nodes", 32);
    if (iterations > 100000 || nodes > 10000)
        throw std::invalid_argument("launch iterations <=100000, graph-nodes <=10000 required");
    graph_owner graph;
    if (op == "graph") {
        const auto setup_start = wall_clock::now();
        CUDA_CHECK(cudaStreamBeginCapture(ctx.stream, cudaStreamCaptureModeThreadLocal));
        for (std::size_t i = 0; i < nodes; ++i) empty_kernel<<<1, 32, 0, ctx.stream>>>();
        CUDA_CHECK(cudaStreamEndCapture(ctx.stream, &graph.graph));
        CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph.executable, graph.graph, 0));
        CUDA_CHECK(cudaGraphUpload(graph.executable, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        r.metrics["graph_setup_ms"] = std::chrono::duration<double, std::milli>(
            wall_clock::now() - setup_start).count();
    }
    auto launch = [&] {
        for (std::size_t i = 0; i < iterations; ++i) {
            if (op == "graph") CUDA_CHECK(cudaGraphLaunch(graph.executable, ctx.stream));
            else empty_kernel<<<1, 32, 0, ctx.stream>>>();
        }
        check_launch();
    };
    r.samples_ms = ctx.measure(launch);
    std::vector<double> submission, completion;
    const auto samples = r.samples_ms.size();
    for (std::size_t i = 0; i < samples; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        const auto start = wall_clock::now();
        launch();
        const auto submitted = wall_clock::now();
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        const auto completed = wall_clock::now();
        submission.push_back(std::chrono::duration<double, std::milli>(submitted - start).count());
        completion.push_back(std::chrono::duration<double, std::milli>(completed - start).count());
    }
    const auto launch_count = static_cast<double>(iterations);
    r.metrics["host_submission_median_ms"] = median_value(submission);
    r.metrics["host_completion_median_ms"] = median_value(completion);
    r.metrics["host_ns_per_api_launch"] = median_value(submission) * 1.0e6 / launch_count;
    const auto submission_stats = analyze_timing(submission), completion_stats = analyze_timing(completion);
    r.metrics["host_sample_count"] = static_cast<double>(samples);
    r.metrics["host_submission_median_us"] = submission_stats.median * 1000;
    r.metrics["host_completion_median_us"] = completion_stats.median * 1000;
    r.metrics["host_submission_min_us"] = submission_stats.minimum * 1000;
    r.metrics["host_submission_max_us"] = submission_stats.maximum * 1000;
    r.metrics["host_completion_min_us"] = completion_stats.minimum * 1000;
    r.metrics["host_completion_max_us"] = completion_stats.maximum * 1000;
    if (samples > 1) {
        r.metrics["host_submission_stddev_us"] = submission_stats.stddev * 1000;
        r.metrics["host_submission_cv"] = submission_stats.cv;
        r.metrics["host_completion_stddev_us"] = completion_stats.stddev * 1000;
        r.metrics["host_completion_cv"] = completion_stats.cv;
    }
    r.metrics["kernels_per_api_launch"] = op == "graph" ? static_cast<double>(nodes) : 1;
    r.work_per_iteration = launch_count;
    r.unit = "Glaunch/s";
    r.metadata["work_domain"] = "launch";
    r.execution = "runtime_api";
    r.validation = "not_applicable";
    r.metadata["validation_scope"] = "successful CUDA completion only; empty kernels have no numerical output";
    r.metadata["timing_scope"] = "device events plus separate host submission and synchronized completion clocks";
    r.metadata["host_sample_scope"] = "separate single-callback host samples; sample count matches device sample count, but host CV does not control stopping";
    r.metadata["operation_count"] = "CUDA API launch calls; graph nodes reported separately";
    return true;
}

struct pinned_owner {
    float* data = nullptr;
    explicit pinned_owner(std::size_t n) { CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), n * sizeof(float))); }
    ~pinned_owner() { if (data) cudaFreeHost(data); }
};

struct managed_owner {
    float* data = nullptr;
    explicit managed_owner(std::size_t n) { CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(&data), n * sizeof(float))); }
    ~managed_owner() { if (data) cudaFree(data); }
};

cudaError_t prefetch_managed(float* pointer, std::size_t bytes, int device,
                             cudaStream_t stream) {
#if CUDART_VERSION >= 13000
    cudaMemLocation location{};
    location.type = device == cudaCpuDeviceId ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
    location.id = device == cudaCpuDeviceId ? 0 : device;
    return cudaMemPrefetchAsync(pointer, bytes, location, 0, stream);
#else
    return cudaMemPrefetchAsync(pointer, bytes, device, stream);
#endif
}

__global__ void managed_increment_kernel(float* data, std::size_t n) {
    const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += step) data[i] += 1.0f;
}

struct peer_allocation {
    int original = 0, peer = 0;
    float* data = nullptr;
    peer_allocation(int original_device, int peer_device, std::size_t n)
        : original(original_device), peer(peer_device) {
        CUDA_CHECK(cudaSetDevice(peer));
        const auto status = cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(float));
        const auto restored = cudaSetDevice(original);
        CUDA_CHECK(status); CUDA_CHECK(restored);
    }
    ~peer_allocation() {
        if (data) {
            cudaSetDevice(peer);
            cudaFree(data);
            cudaSetDevice(original);
        }
    }
};

bool run_transfer(context& ctx, result& r) {
    const auto op = ctx.opts.get("op", "h2d");
    const auto dtype = ctx.opts.get("dtype", "f32");
    if (dtype != "f32") { skip(r, "transfer uses f32 payloads; byte bandwidth is type-independent"); return true; }
    const auto n = positive_size(ctx.opts, "elements", 1 << 22);
    const auto bytes = checked_product(n, sizeof(float));
    ctx.memory_guard(checked_product(bytes, 2));
    r.execution = "runtime_api";
    r.unit = "GB/s";
    r.metadata["work_domain"] = "data_movement";
    r.bytes_per_iteration = static_cast<double>(bytes);
    r.metadata["bandwidth_scope"] = "one-way useful payload bytes, not simultaneous bus read+write bytes";
    if (op == "managed_prefetch" || op == "managed_first_touch") {
        if (!ctx.prop.managedMemory || !ctx.prop.concurrentManagedAccess) {
            skip(r, "managed migration benchmark requires managedMemory and concurrentManagedAccess");
            return true;
        }
        managed_owner allocation(n);
        const auto expected = float_input(n);
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        const int threads = thread_count(ctx), blocks = grid_count(ctx, n, threads);
        auto prepare = [&] {
            // Evict to CPU and touch every element before EACH sample, outside timing.
            CUDA_CHECK(prefetch_managed(allocation.data, bytes, cudaCpuDeviceId, ctx.stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
            std::copy(expected.begin(), expected.end(), allocation.data);
        };
        auto launch = [&] {
            if (op == "managed_prefetch")
                CUDA_CHECK(prefetch_managed(allocation.data, bytes, device, ctx.stream));
            else {
                managed_increment_kernel<<<blocks, threads, 0, ctx.stream>>>(allocation.data, n);
                check_launch();
            }
        };
        r.samples_ms = measure_host_completed(ctx, prepare, launch);
        CUDA_CHECK(prefetch_managed(allocation.data, bytes, cudaCpuDeviceId, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        begin_validation(r, n);
        for (std::size_t i = 0; i < n; ++i)
            check_value(r, allocation.data[i], expected[i] + (op == "managed_first_touch" ? 1.0f : 0.0f));
        r.metadata["timing_clock"] = "host steady_clock through stream synchronization";
        r.metadata["timing"] = r.metadata["timing_clock"];
        r.metadata["timing_scope"] = op == "managed_prefetch" ?
            "CPU-to-GPU prefetch after CPU migration and full host write" :
            "GPU increment kernel plus demand-page migration after CPU migration and full host write";
        return true;
    }
    if (op == "p2p") {
        int original, count;
        CUDA_CHECK(cudaGetDevice(&original)); CUDA_CHECK(cudaGetDeviceCount(&count));
        const auto peer64 = ctx.opts.integer("peer", original == 0 ? 1 : 0);
        if (peer64 < 0 || peer64 >= count || peer64 == original) {
            skip(r, "peer must name a different visible CUDA device"); return true;
        }
        const int peer = static_cast<int>(peer64);
        int accessible = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, original, peer));
        if (!accessible) { skip(r, "cudaDeviceCanAccessPeer reports no P2P path"); return true; }
        const auto enable_status = cudaDeviceEnablePeerAccess(peer, 0);
        if (enable_status != cudaSuccess && enable_status != cudaErrorPeerAccessAlreadyEnabled)
            CUDA_CHECK(enable_status);
        if (enable_status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        device_buffer<float> source(n);
        peer_allocation destination(original, peer, n);
        const auto expected = float_input(n);
        upload(ctx, source.get(), expected);
        auto launch = [&] {
            CUDA_CHECK(cudaMemcpyPeerAsync(destination.data, peer, source.get(), original, bytes, ctx.stream));
        };
        r.samples_ms = ctx.measure(launch);
        std::vector<float> got(n);
        CUDA_CHECK(cudaSetDevice(peer));
        const auto copy_status = cudaMemcpy(got.data(), destination.data, bytes, cudaMemcpyDeviceToHost);
        const auto restore_status = cudaSetDevice(original);
        CUDA_CHECK(copy_status); CUDA_CHECK(restore_status);
        begin_validation(r, n);
        for (std::size_t i = 0; i < n; ++i) check_value(r, got[i], expected[i]);
        if (enable_status == cudaSuccess) CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
        r.metrics["peer_device"] = peer;
        r.metadata["interconnect"] = "runtime P2P path; no NVLink/PCIe classification inferred";
        return true;
    }
    if (op != "h2d" && op != "d2h" && op != "d2d") {
        skip(r, "unknown transfer operation: " + op); return true;
    }
    const bool pinned = ctx.opts.flag("pinned", true);
    std::vector<float> pageable = float_input(n);
    std::unique_ptr<pinned_owner> pinned_data;
    float* host = pageable.data();
    if (pinned && op != "d2d") {
        pinned_data.reset(new pinned_owner(n));
        host = pinned_data->data;
        std::copy(pageable.begin(), pageable.end(), host);
    }
    device_buffer<float> source(n), destination(n);
    upload(ctx, source.get(), pageable);
    auto launch = [&] {
        if (op == "h2d")
            CUDA_CHECK(cudaMemcpyAsync(destination.get(), host, bytes, cudaMemcpyHostToDevice, ctx.stream));
        else if (op == "d2h")
            CUDA_CHECK(cudaMemcpyAsync(host, source.get(), bytes, cudaMemcpyDeviceToHost, ctx.stream));
        else
            CUDA_CHECK(cudaMemcpyAsync(destination.get(), source.get(), bytes, cudaMemcpyDeviceToDevice, ctx.stream));
    };
    if (pinned || op == "d2d") {
        r.samples_ms = ctx.measure(launch);
        r.metadata["timing_clock"] = op == "d2d" ? "CUDA events; device-to-device transfer" :
            "CUDA events; pageable staging excluded by pinned allocation";
    } else {
        // Pageable staging can block on the host. Device events alone undercount it.
        r.samples_ms = measure_host_completed(ctx, [] {}, launch);
        r.metadata["timing_clock"] = "host steady_clock including pageable staging and completion";
    }
    r.metadata["timing"] = r.metadata["timing_clock"];
    begin_validation(r, n);
    if (op == "d2h") {
        for (std::size_t i = 0; i < n; ++i) check_value(r, host[i], input_value(i));
    } else {
        const auto got = download(ctx, destination.get(), n);
        for (std::size_t i = 0; i < n; ++i) check_value(r, got[i], input_value(i));
    }
    r.metadata["host_memory"] = op == "d2d" ? "not_applicable" : (pinned ? "pinned" : "pageable");
    return true;
}

#if defined(ARCHBENCH_HAS_CUSPARSE) && ARCHBENCH_HAS_CUSPARSE
void sparse_check(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuSPARSE: ") + cusparseGetErrorString(status));
}

struct sparse_owner {
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matrix = nullptr;
    cusparseDnVecDescr_t x = nullptr, y = nullptr;
    cusparseDnMatDescr_t b = nullptr, c = nullptr;
    ~sparse_owner() {
        if (c) cusparseDestroyDnMat(c);
        if (b) cusparseDestroyDnMat(b);
        if (y) cusparseDestroyDnVec(y);
        if (x) cusparseDestroyDnVec(x);
        if (matrix) cusparseDestroySpMat(matrix);
        if (handle) cusparseDestroy(handle);
    }
};

template <class T>
void sparse_case(context& ctx, result& r, const std::string& op) {
    const auto rows = positive_size(ctx.opts, "rows", 65536);
    const auto cols = positive_size(ctx.opts, "cols", 65536);
    const auto per_row = positive_size(ctx.opts, "nnz-per-row", 16);
    const auto nnz = checked_product(rows, per_row);
    const auto columns = op == "csr_spmm" ? positive_size(ctx.opts, "n", 64) : 1;
    if (per_row > cols || nnz > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        cols > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        rows > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("CSR uses int32 offsets/indices and nnz-per-row <= cols");
    const auto x_count = checked_product(cols, columns), y_count = checked_product(rows, columns);
    const auto base_bytes = checked_product(nnz, sizeof(T) + sizeof(int)) +
                            checked_product(rows + 1, sizeof(int)) +
                            checked_product(x_count + y_count, sizeof(T));
    ctx.memory_guard(base_bytes);
    std::vector<int> row_offsets(rows + 1), indices(nnz);
    std::vector<T> values(nnz);
    for (std::size_t row = 0; row < rows; ++row) {
        row_offsets[row] = static_cast<int>(row * per_row);
        for (std::size_t j = 0; j < per_row; ++j)
            indices[row * per_row + j] = static_cast<int>(((row * 13) % cols + j) % cols);
        std::sort(indices.begin() + row * per_row, indices.begin() + (row + 1) * per_row);
        for (std::size_t j = 0; j < per_row; ++j)
            values[row * per_row + j] = static_cast<T>(input_value(row * per_row + j, 7));
    }
    row_offsets[rows] = static_cast<int>(nnz);
    std::vector<T> x(x_count);
    for (std::size_t i = 0; i < x_count; ++i) x[i] = static_cast<T>(input_value(i, 3));
    device_buffer<int> offsets_device(rows + 1), indices_device(nnz);
    device_buffer<T> values_device(nnz), x_device(x_count), y_device(y_count);
    upload(ctx, offsets_device.get(), row_offsets); upload(ctx, indices_device.get(), indices);
    upload(ctx, values_device.get(), values); upload(ctx, x_device.get(), x);
    const auto dtype = std::is_same<T, double>::value ? CUDA_R_64F : CUDA_R_32F;
    const T alpha = 1, beta = 0;
    sparse_owner owner;
    sparse_check(cusparseCreate(&owner.handle));
    int library_version = 0;
    sparse_check(cusparseGetVersion(owner.handle, &library_version));
    r.metadata["cusparse_version"] = std::to_string(library_version);
    sparse_check(cusparseSetStream(owner.handle, ctx.stream));
    sparse_check(cusparseCreateCsr(&owner.matrix, static_cast<int64_t>(rows), static_cast<int64_t>(cols),
        static_cast<int64_t>(nnz), offsets_device.get(), indices_device.get(), values_device.get(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, dtype));
    std::size_t temporary_bytes = 0;
    if (op == "csr_spmv") {
        sparse_check(cusparseCreateDnVec(&owner.x, static_cast<int64_t>(cols), x_device.get(), dtype));
        sparse_check(cusparseCreateDnVec(&owner.y, static_cast<int64_t>(rows), y_device.get(), dtype));
        sparse_check(cusparseSpMV_bufferSize(owner.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, owner.matrix, owner.x, &beta, owner.y, dtype, CUSPARSE_SPMV_ALG_DEFAULT, &temporary_bytes));
    } else {
        sparse_check(cusparseCreateDnMat(&owner.b, static_cast<int64_t>(cols), static_cast<int64_t>(columns),
            static_cast<int64_t>(cols), x_device.get(), dtype, CUSPARSE_ORDER_COL));
        sparse_check(cusparseCreateDnMat(&owner.c, static_cast<int64_t>(rows), static_cast<int64_t>(columns),
            static_cast<int64_t>(rows), y_device.get(), dtype, CUSPARSE_ORDER_COL));
        sparse_check(cusparseSpMM_bufferSize(owner.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, owner.matrix, owner.b, &beta, owner.c,
            dtype, CUSPARSE_SPMM_ALG_DEFAULT, &temporary_bytes));
    }
    ctx.memory_guard(base_bytes + temporary_bytes);
    device_buffer<unsigned char> temporary(std::max<std::size_t>(temporary_bytes, 1));
    if (ctx.opts.flag("preprocess", true)) {
        const auto start = wall_clock::now();
        if (op == "csr_spmv")
            sparse_check(cusparseSpMV_preprocess(owner.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, owner.matrix, owner.x, &beta, owner.y, dtype, CUSPARSE_SPMV_ALG_DEFAULT, temporary.get()));
        else
            sparse_check(cusparseSpMM_preprocess(owner.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, owner.matrix, owner.b, &beta, owner.c,
                dtype, CUSPARSE_SPMM_ALG_DEFAULT, temporary.get()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        r.metrics["preprocess_ms"] = std::chrono::duration<double, std::milli>(wall_clock::now() - start).count();
    }
    auto launch = [&] {
        if (op == "csr_spmv")
            sparse_check(cusparseSpMV(owner.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                owner.matrix, owner.x, &beta, owner.y, dtype, CUSPARSE_SPMV_ALG_DEFAULT, temporary.get()));
        else
            sparse_check(cusparseSpMM(owner.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, owner.matrix, owner.b, &beta, owner.c,
                dtype, CUSPARSE_SPMM_ALG_DEFAULT, temporary.get()));
    };
    r.samples_ms = ctx.measure(launch);
    const auto got = download(ctx, y_device.get(), y_count);
    begin_validation(r, y_count);
    const auto atol = std::is_same<T, double>::value ? 1.0e-11L : 1.0e-4L;
    const auto rtol = std::is_same<T, double>::value ? 1.0e-11L : 1.0e-5L;
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t row = 0; row < rows; ++row) {
            long double reference = 0;
            for (int offset = row_offsets[row]; offset < row_offsets[row + 1]; ++offset)
                reference += static_cast<long double>(values[offset]) * x[column * cols + indices[offset]];
            check_value(r, got[column * rows + row], reference, atol, rtol);
        }
    }
    r.execution = "library_selected";
    r.unit = "Gflop/s";
    r.metadata["work_domain"] = "floating_point";
    r.work_per_iteration = 2.0 * nnz * columns;
    r.bytes_per_iteration = static_cast<double>(base_bytes);
    r.metrics["nonzeros"] = static_cast<double>(nnz);
    r.metrics["density"] = static_cast<double>(per_row) / cols;
    r.metrics["temporary_storage_bytes"] = static_cast<double>(temporary_bytes);
    r.metadata["sparsity"] = "general CSR; not 2:4 structured sparsity or sparse Tensor Core MMA";
    r.metadata["library"] = "cuSPARSE";
    r.metadata["algorithm"] = "CUSPARSE_SPMV_ALG_DEFAULT or CUSPARSE_SPMM_ALG_DEFAULT";
    r.metadata["bandwidth_scope"] = "logical CSR and dense operand sizes; physical reuse is not inferred";
}
#endif

bool run_sparse(context& ctx, result& r) {
#if defined(ARCHBENCH_HAS_CUSPARSE) && ARCHBENCH_HAS_CUSPARSE
    const auto op = ctx.opts.get("op", "csr_spmv"), dtype = ctx.opts.get("dtype", "f32");
    if (op != "csr_spmv" && op != "csr_spmm") { skip(r, "sparse operations are csr_spmv, csr_spmm"); return true; }
    if (dtype == "f32") sparse_case<float>(ctx, r, op);
    else if (dtype == "f64") sparse_case<double>(ctx, r, op);
    else skip(r, "CSR baseline supports f32/f64");
#else
    (void)ctx;
    skip(r, "cuSPARSE backend was not enabled in this build", "missing_dependency");
#endif
    return true;
}

#if defined(ARCHBENCH_HAS_CUFFT) && ARCHBENCH_HAS_CUFFT
void fft_check(cufftResult status) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error("cuFFT status " + std::to_string(static_cast<int>(status)));
}

struct fft_owner {
    cufftHandle plan = 0;
    ~fft_owner() { if (plan) cufftDestroy(plan); }
};

template <class T> struct fft_types;
template <> struct fft_types<float> {
    using complex = cufftComplex;
    static constexpr cufftType c2c = CUFFT_C2C, r2c = CUFFT_R2C, c2r = CUFFT_C2R;
};
template <> struct fft_types<double> {
    using complex = cufftDoubleComplex;
    static constexpr cufftType c2c = CUFFT_Z2Z, r2c = CUFFT_D2Z, c2r = CUFFT_Z2D;
};
template <> struct fft_types<__half> {
    struct alignas(4) complex { __half x, y; };
};
template <> struct fft_types<__nv_bfloat16> {
    struct alignas(4) complex { __nv_bfloat16 x, y; };
};

std::vector<std::size_t> coordinates(std::size_t index, const std::vector<int>& dims) {
    std::vector<std::size_t> out(dims.size());
    for (std::size_t rev = dims.size(); rev-- > 0;) {
        out[rev] = index % static_cast<std::size_t>(dims[rev]);
        index /= static_cast<std::size_t>(dims[rev]);
    }
    return out;
}

template <class T>
typename fft_types<T>::complex hermitian_value(std::size_t index,
                                               const std::vector<int>& dims, unsigned salt,
                                               float scale = 1.0f) {
    const auto coordinate = coordinates(index, dims);
    std::size_t partner = 0;
    for (std::size_t d = 0; d < dims.size(); ++d)
        partner = partner * dims[d] + (dims[d] - coordinate[d]) % dims[d];
    const auto canonical = std::min(index, partner);
    typename fft_types<T>::complex value;
    value.x = static_cast<T>(input_value(canonical, salt) * scale);
    value.y = index == partner ? T(0) :
        static_cast<T>(input_value(canonical, salt + 3) * scale * (index < partner ? 1.0f : -1.0f));
    return value;
}

template <class T>
void fft_case(context& ctx, result& r, const std::string& op) {
    using C = typename fft_types<T>::complex;
    constexpr bool narrow = std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value;
    static_assert(sizeof(C) == 2 * sizeof(T), "FFT complex data must be interleaved without padding");
    const auto rank64 = ctx.opts.integer("rank", 1);
    if (rank64 < 1 || rank64 > 3) throw std::invalid_argument("FFT rank must be 1..3");
    const int rank = static_cast<int>(rank64);
    const auto nx = positive_size(ctx.opts, "nx", ctx.opts.integer("n", 4096));
    const auto ny = positive_size(ctx.opts, "ny", 16), nz = positive_size(ctx.opts, "nz", 16);
    const auto batch = positive_size(ctx.opts, "batch", 1);
    if (nx > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        ny > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        nz > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        batch > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("FFT dimensions/batch exceed 32-bit cuFFT plan limits");
    std::vector<int> dims;
    if (rank == 3) dims.push_back(static_cast<int>(nz));
    if (rank >= 2) dims.push_back(static_cast<int>(ny));
    dims.push_back(static_cast<int>(nx));
    std::size_t volume = 1;
    for (int dimension : dims) volume = checked_product(volume, static_cast<std::size_t>(dimension));
    if (volume > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("FFT baseline uses a 32-bit transform distance");
    if constexpr (narrow) {
        for (int dimension : dims) {
            if (dimension & (dimension - 1)) {
                skip(r, "cuFFT half/bfloat16 transforms require power-of-two dimensions"); return;
            }
        }
    }
    const float input_scale = narrow ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(volume))) : 1.0f;
    const auto packed_volume = volume / nx * (nx / 2 + 1);
    const auto real_count = checked_product(volume, batch);
    const auto packed_count = checked_product(packed_volume, batch);
    const auto complex_count = op == "c2c" ? real_count : packed_count;
    const auto input_bytes = op == "r2c" ? checked_product(real_count, sizeof(T)) :
                                                checked_product(complex_count, sizeof(C));
    const auto output_bytes = op == "c2r" ? checked_product(real_count, sizeof(T)) :
                                                 checked_product(complex_count, sizeof(C));
    ctx.memory_guard(input_bytes + output_bytes);
    device_buffer<unsigned char> input(input_bytes), output(output_bytes);
    std::vector<T> real_input;
    std::vector<C> complex_input;
    if (op == "r2c") {
        real_input.resize(real_count);
        for (std::size_t i = 0; i < real_count; ++i) real_input[i] = static_cast<T>(input_value(i, 11) * input_scale);
        upload(ctx, reinterpret_cast<T*>(input.get()), real_input);
    } else {
        complex_input.resize(complex_count);
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t i = 0; i < (op == "c2c" ? volume : packed_volume); ++i) {
                auto& value = complex_input[b * (op == "c2c" ? volume : packed_volume) + i];
                if (op == "c2c") {
                    value.x = static_cast<T>(input_value(b * volume + i, 5) * input_scale);
                    value.y = static_cast<T>(input_value(b * volume + i, 19) * input_scale);
                } else {
                    const auto full_index = i / (nx / 2 + 1) * nx + i % (nx / 2 + 1);
                    value = hermitian_value<T>(full_index, dims, static_cast<unsigned>(b + 5), input_scale);
                }
            }
        }
        upload(ctx, reinterpret_cast<C*>(input.get()), complex_input);
    }
    const auto requested_direction = ctx.opts.get("direction", op == "c2r" ? "inverse" : "forward");
    if (requested_direction != "forward" && requested_direction != "inverse")
        throw std::invalid_argument("FFT direction must be forward or inverse");
    const int direction = requested_direction == "inverse" ? CUFFT_INVERSE : CUFFT_FORWARD;
    if ((op == "r2c" && direction != CUFFT_FORWARD) || (op == "c2r" && direction != CUFFT_INVERSE)) {
        skip(r, "r2c requires --direction forward; c2r requires --direction inverse"); return;
    }
    fft_owner owner;
    const auto plan_start = wall_clock::now();
    fft_check(cufftCreate(&owner.plan));
    int library_version = 0;
    fft_check(cufftGetVersion(&library_version));
    r.metadata["cufft_version"] = std::to_string(library_version);
    fft_check(cufftSetAutoAllocation(owner.plan, 0));
    std::size_t workspace = 0;
    const int input_distance = static_cast<int>(op == "c2r" ? packed_volume : volume);
    const int output_distance = static_cast<int>(op == "r2c" ? packed_volume : volume);
    if constexpr (narrow) {
        const auto real_type = std::is_same<T, __half>::value ? CUDA_R_16F : CUDA_R_16BF;
        const auto complex_type = std::is_same<T, __half>::value ? CUDA_C_16F : CUDA_C_16BF;
        std::vector<long long> dims64(dims.begin(), dims.end());
        fft_check(cufftXtMakePlanMany(owner.plan, rank, dims64.data(), nullptr, 1, input_distance,
            op == "r2c" ? real_type : complex_type, nullptr, 1, output_distance,
            op == "c2r" ? real_type : complex_type, static_cast<long long>(batch), &workspace, complex_type));
    } else {
        const auto type = op == "c2c" ? fft_types<T>::c2c : (op == "r2c" ? fft_types<T>::r2c : fft_types<T>::c2r);
        fft_check(cufftMakePlanMany(owner.plan, rank, dims.data(), nullptr, 1, input_distance,
            nullptr, 1, output_distance, type, static_cast<int>(batch), &workspace));
    }
    ctx.memory_guard(input_bytes + output_bytes + workspace);
    device_buffer<unsigned char> work(std::max<std::size_t>(workspace, 1));
    fft_check(cufftSetWorkArea(owner.plan, work.get()));
    fft_check(cufftSetStream(owner.plan, ctx.stream));
    r.metrics["plan_setup_ms"] = std::chrono::duration<double, std::milli>(wall_clock::now() - plan_start).count();
    // C2R transforms may overwrite the input even out of place. Preserve it with
    // a same-sized immutable buffer and explicitly include restoration in timing.
    std::unique_ptr<device_buffer<unsigned char>> inverse_original;
    if (op == "c2r") {
        ctx.memory_guard(input_bytes * 2 + output_bytes + workspace);
        inverse_original.reset(new device_buffer<unsigned char>(input_bytes));
        CUDA_CHECK(cudaMemcpyAsync(inverse_original->get(), input.get(), input_bytes,
                                   cudaMemcpyDeviceToDevice, ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    }
    auto launch = [&] {
        if (op == "c2r")
            CUDA_CHECK(cudaMemcpyAsync(input.get(), inverse_original->get(), input_bytes,
                                       cudaMemcpyDeviceToDevice, ctx.stream));
        if constexpr (narrow) {
            fft_check(cufftXtExec(owner.plan, input.get(), output.get(), direction));
        } else if constexpr (std::is_same<T, float>::value) {
            if (op == "c2c") fft_check(cufftExecC2C(owner.plan, reinterpret_cast<cufftComplex*>(input.get()),
                reinterpret_cast<cufftComplex*>(output.get()), direction));
            else if (op == "r2c") fft_check(cufftExecR2C(owner.plan, reinterpret_cast<cufftReal*>(input.get()),
                reinterpret_cast<cufftComplex*>(output.get())));
            else fft_check(cufftExecC2R(owner.plan, reinterpret_cast<cufftComplex*>(input.get()),
                reinterpret_cast<cufftReal*>(output.get())));
        } else {
            if (op == "c2c") fft_check(cufftExecZ2Z(owner.plan, reinterpret_cast<cufftDoubleComplex*>(input.get()),
                reinterpret_cast<cufftDoubleComplex*>(output.get()), direction));
            else if (op == "r2c") fft_check(cufftExecD2Z(owner.plan, reinterpret_cast<cufftDoubleReal*>(input.get()),
                reinterpret_cast<cufftDoubleComplex*>(output.get())));
            else fft_check(cufftExecZ2D(owner.plan, reinterpret_cast<cufftDoubleComplex*>(input.get()),
                reinterpret_cast<cufftDoubleReal*>(output.get())));
        }
    };
    r.samples_ms = ctx.measure(launch);
    const auto output_count = op == "c2r" ? real_count : complex_count;
    const auto sample_count = std::min(output_count, positive_size(ctx.opts, "check-bins", 32));
    std::vector<T> real_output;
    std::vector<C> complex_output;
    if (op == "c2r") real_output = download(ctx, reinterpret_cast<T*>(output.get()), real_count);
    else complex_output = download(ctx, reinterpret_cast<C*>(output.get()), complex_count);
    begin_validation(r, sample_count);
    r.metadata["validation_scope"] = sample_count == output_count ?
        "all_output_elements_independent_long_double_DFT" : "evenly_spaced_output_elements_independent_long_double_DFT";
    const long double pi = std::acos(-1.0L);
    const long double atol = narrow ? (std::is_same<T, __half>::value ? 0.004L : 0.025L) *
        std::max(1.0L, std::sqrt(std::log2(static_cast<long double>(volume)))) : std::is_same<T, double>::value ?
        1.0e-10L * std::sqrt(static_cast<long double>(volume)) :
        3.0e-5L * std::sqrt(static_cast<long double>(volume));
    const long double rtol = narrow ? (std::is_same<T, __half>::value ? 0.01L : 0.06L) :
        std::is_same<T, double>::value ? 1.0e-10L : 2.0e-4L;
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        const auto index = sample_count == 1 ? 0 : sample * (output_count - 1) / (sample_count - 1);
        const auto out_volume = op == "r2c" ? packed_volume : volume;
        const auto b = index / out_volume;
        const auto compact_index = index % out_volume;
        const auto full_index = op == "r2c" ?
            compact_index / (nx / 2 + 1) * nx + compact_index % (nx / 2 + 1) : compact_index;
        const auto frequency = coordinates(full_index, dims);
        std::complex<long double> reference(0, 0);
        for (std::size_t j = 0; j < volume; ++j) {
            const auto position = coordinates(j, dims);
            long double turns = 0;
            for (std::size_t d = 0; d < dims.size(); ++d)
                turns += static_cast<long double>(frequency[d]) * position[d] / dims[d];
            const auto angle = (op == "c2r" || direction == CUFFT_INVERSE ? 2 : -2) * pi * turns;
            std::complex<long double> value;
            if (op == "r2c") value = static_cast<long double>(load_value(real_input[b * volume + j]));
            else if (op == "c2c") value = {static_cast<long double>(load_value(complex_input[b * volume + j].x)),
                                           static_cast<long double>(load_value(complex_input[b * volume + j].y))};
            else {
                const auto generated = hermitian_value<T>(j, dims, static_cast<unsigned>(b + 5), input_scale);
                value = {static_cast<long double>(load_value(generated.x)), static_cast<long double>(load_value(generated.y))};
            }
            reference += value * std::complex<long double>(std::cos(angle), std::sin(angle));
        }
        if (op == "c2r") check_value(r, load_value(real_output[index]), reference.real(), atol, rtol);
        else {
            check_value(r, load_value(complex_output[index].x), reference.real(), atol, rtol);
            check_value(r, load_value(complex_output[index].y), reference.imag(), atol, rtol);
        }
    }
    r.execution = "library_selected";
    r.unit = "Gitem/s";
    r.metadata["work_domain"] = "item";
    r.work_per_iteration = static_cast<double>(real_count);
    r.bytes_per_iteration = static_cast<double>(input_bytes + output_bytes + (op == "c2r" ? 2 * input_bytes : 0));
    r.metrics["temporary_storage_bytes"] = static_cast<double>(workspace);
    r.metrics["transform_points"] = static_cast<double>(volume);
    r.metrics["validation_atol"] = static_cast<double>(atol);
    r.metrics["validation_rtol"] = static_cast<double>(rtol);
    r.metrics["input_scale"] = input_scale;
    r.metrics["estimated_fft_flops"] = (op == "c2c" ? 5.0 : 2.5) * batch * volume * std::log2(static_cast<double>(volume));
    r.metadata["operation_count"] = "input transform points; estimated FFT FLOPs are a separate model";
    r.metadata["normalization"] = "cuFFT unnormalized; inverse is not divided by transform volume";
    r.metadata["library"] = "cuFFT";
    if (op == "c2r") r.metadata["timing_scope"] = "input restoration D2D copy plus inverse FFT; input may be overwritten by cuFFT";
}
#endif

bool run_fft(context& ctx, result& r) {
#if defined(ARCHBENCH_HAS_CUFFT) && ARCHBENCH_HAS_CUFFT
    const auto op = ctx.opts.get("op", "c2c"), dtype = ctx.opts.get("dtype", "c64");
    if (op != "c2c" && op != "r2c" && op != "c2r") { skip(r, "FFT operations are c2c, r2c, c2r"); return true; }
    if (dtype == "f32" || dtype == "c64") fft_case<float>(ctx, r, op);
    else if (dtype == "f64" || dtype == "c128") fft_case<double>(ctx, r, op);
    else if (dtype == "f16") fft_case<__half>(ctx, r, op);
    else if (dtype == "bf16") fft_case<__nv_bfloat16>(ctx, r, op);
    else skip(r, "cuFFT supports f16, bf16, f32/c64, f64/c128 in this backend");
#else
    (void)ctx;
    skip(r, "cuFFT backend was not enabled in this build", "missing_dependency");
#endif
    return true;
}

} // namespace

bool run_systems(context& ctx, result& r) {
    const auto family = ctx.opts.get("family");
    if (family == "fft" || family == "sparse") {
        const std::string library = family == "fft" ? "cufft" : "cusparse";
        const auto backend = ctx.opts.get("backend", library);
        if (backend != library && backend != "native") {
            skip(r, "unsupported " + family + " backend: " + backend +
                    "; use " + library + ", native, software or emulated", "not_implemented");
            return true;
        }
    }
    if (family == "memory") return run_memory(ctx, r);
    if (family == "atomic") return run_atomic(ctx, r);
    if (family == "collective") return run_collective(ctx, r);
    if (family == "workload") return run_workload(ctx, r);
    if (family == "launch") return run_launch(ctx, r);
    if (family == "transfer") return run_transfer(ctx, r);
    if (family == "sparse") return run_sparse(ctx, r);
    if (family == "fft") return run_fft(ctx, r);
    return false;
}

} // namespace archbench

