// SPDX-License-Identifier: AGPL-3.0-or-later
// Portable CUDA algorithm baselines: no cuSPARSE/cuFFT calls and no CPU timing
// presented as GPU work. See docs/systems_software.md for comparison boundaries.
#include "archbench/common.cuh"
#include "archbench/systems_software_model.hpp"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace archbench {
namespace {
namespace swm = systems_software_model;

std::size_t sw_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("software baseline size product overflow");
    return a * b;
}

std::size_t sw_sum(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b)
        throw std::invalid_argument("software baseline size sum overflow");
    return a + b;
}

std::size_t sw_positive(const options& opts, const char* key, int64_t fallback) {
    const int64_t value = opts.integer(key, fallback);
    if (value <= 0) throw std::invalid_argument(std::string(key) + " must be positive");
    return static_cast<std::size_t>(value);
}

int sw_threads(const context& ctx) {
    const auto value = ctx.opts.integer("threads", 256);
    if (value < 32 || value > ctx.prop.maxThreadsPerBlock || value % 32)
        throw std::invalid_argument("software kernel threads must be a supported warp multiple");
    return static_cast<int>(value);
}

int sw_blocks(const context& ctx, std::size_t count, int threads) {
    const auto value = ctx.opts.integer("blocks", std::max(1, ctx.prop.multiProcessorCount * 4));
    if (value < 1 || value > ctx.prop.maxGridSize[0]) throw std::invalid_argument("invalid software kernel grid");
    return static_cast<int>(std::min<std::size_t>(value, std::max<std::size_t>(1, (count + threads - 1) / threads)));
}

template<class T> void sw_upload(context& ctx, T* output, const std::vector<T>& input) {
    CUDA_CHECK(cudaMemcpyAsync(output, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice, ctx.stream));
}

template<class T> std::vector<T> sw_download(context& ctx, const T* input, std::size_t count) {
    std::vector<T> output(count);
    CUDA_CHECK(cudaMemcpyAsync(output.data(), input, count * sizeof(T), cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    return output;
}

std::string sw_number(long double value) {
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<long double>::max_digits10) << value;
    return output.str();
}

void sw_check(result& r, long double got, long double expected, long double atol,
              long double rtol, std::size_t index) {
    const auto error = std::abs(got - expected);
    const auto relative = error / std::max(std::abs(expected), 1e-30L);
    const auto unavailable = std::numeric_limits<double>::infinity();
    r.max_abs_error = std::max(r.max_abs_error, std::isfinite(error) ? double(error) : unavailable);
    r.max_rel_error = std::max(r.max_rel_error, std::isfinite(relative) ? double(relative) : unavailable);
    if (!std::isfinite(got) || !std::isfinite(expected) || error > atol + rtol * std::abs(expected)) {
        if (r.validation != "failed") {
            r.metadata["first_failure_index"] = std::to_string(index);
            r.metadata["first_failure_observed"] = sw_number(got);
            r.metadata["first_failure_expected"] = sw_number(expected);
        }
        r.status = "validation_failed";
        r.validation = "failed";
        r.reason = "software CUDA output disagrees with the independent CPU reference";
    }
}

template<class T> __global__ void sw_csr_kernel(const int* offsets, const int* indices,
        const T* values, const T* x, T* y, std::size_t rows, std::size_t cols,
        std::size_t output_count) {
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < output_count; index += std::size_t(blockDim.x) * gridDim.x) {
        const auto row = index % rows, column = index / rows;
        T value = 0;
        for (int entry = offsets[row]; entry < offsets[row + 1]; ++entry) {
            if constexpr (std::is_same<T, double>::value)
                value = fma(values[entry], x[column * cols + indices[entry]], value);
            else value = fmaf(values[entry], x[column * cols + indices[entry]], value);
        }
        y[index] = value;
    }
}

template<class T> void sw_sparse(context& ctx, result& r, const std::string& op) {
    const auto rows = sw_positive(ctx.opts, "rows", 65536), cols = sw_positive(ctx.opts, "cols", 65536);
    const auto per_row = sw_positive(ctx.opts, "nnz-per-row", 16);
    const auto nnz = sw_product(rows, per_row);
    const auto columns = op == "csr_spmm" ? sw_positive(ctx.opts, "n", 64) : 1;
    if (per_row > cols || nnz > INT32_MAX || rows > INT32_MAX || cols > INT32_MAX)
        throw std::invalid_argument("CSR uses int32 offsets/indices and nnz-per-row <= cols");
    const auto x_count = sw_product(cols, columns), y_count = sw_product(rows, columns);
    const auto bytes = sw_sum(sw_sum(sw_product(nnz, sizeof(T) + sizeof(int)), sw_product(rows + 1, sizeof(int))),
                              sw_product(sw_sum(x_count, y_count), sizeof(T)));
    ctx.memory_guard(bytes);
    std::vector<int> offsets(rows + 1), indices(nnz);
    std::vector<T> values(nnz), x(x_count);
    for (std::size_t row = 0; row < rows; ++row) {
        offsets[row] = static_cast<int>(row * per_row);
        for (std::size_t j = 0; j < per_row; ++j)
            indices[row * per_row + j] = static_cast<int>(((row * 13) % cols + j) % cols);
        std::sort(indices.begin() + row * per_row, indices.begin() + (row + 1) * per_row);
        for (std::size_t j = 0; j < per_row; ++j) values[row * per_row + j] = static_cast<T>(swm::input_value(row * per_row + j, 7));
    }
    offsets[rows] = static_cast<int>(nnz);
    for (std::size_t i = 0; i < x_count; ++i) x[i] = static_cast<T>(swm::input_value(i, 3));
    device_buffer<int> dr(rows + 1), di(nnz);
    device_buffer<T> dv(nnz), dx(x_count), dy(y_count);
    sw_upload(ctx, dr.get(), offsets); sw_upload(ctx, di.get(), indices);
    sw_upload(ctx, dv.get(), values); sw_upload(ctx, dx.get(), x);
    const int threads = sw_threads(ctx), blocks = sw_blocks(ctx, y_count, threads);
    auto launch = [&] { sw_csr_kernel<<<blocks, threads, 0, ctx.stream>>>(dr.get(), di.get(), dv.get(), dx.get(), dy.get(), rows, cols, y_count); };
    r.samples_ms = ctx.measure(launch);
    const auto actual = sw_download(ctx, dy.get(), y_count);
    r.validation = "passed";
    const long double atol = std::is_same<T, double>::value ? 1e-11L : 1e-4L;
    const long double rtol = std::is_same<T, double>::value ? 1e-11L : 1e-5L;
    for (std::size_t column = 0; column < columns; ++column) {
        for (std::size_t row = 0; row < rows; ++row) {
            long double reference = 0;
            for (int entry = offsets[row]; entry < offsets[row + 1]; ++entry)
                reference += static_cast<long double>(values[entry]) * x[column * cols + indices[entry]];
            sw_check(r, actual[column * rows + row], reference, atol, rtol, column * rows + row);
        }
    }
    r.execution = "software_cuda_csr";
    r.unit = "Gflop/s";
    r.work_per_iteration = 2.0 * nnz * columns;
    r.bytes_per_iteration = static_cast<double>(bytes);
    r.metadata["work_domain"] = "floating_point";
    r.metadata["algorithm"] = "one CUDA thread per output element; sequential CSR row dot product";
    r.metadata["library"] = "none";
    r.metadata["input_contract"] = "same stored values, sorted int32 CSR columns and column-major dense operands as the cuSPARSE baseline";
    r.metadata["validation_scope"] = "all output elements against independent long-double CSR products";
    r.metadata["timing_scope"] = "GPU CSR multiplication only; host input generation and validation excluded";
    r.metadata["preprocess"] = "not required by the software CSR algorithm; input sorting is setup work";
    r.metadata["sparsity"] = "general CSR, not structured sparse Tensor Core arithmetic";
    r.metrics["nonzeros"] = static_cast<double>(nnz);
    r.metrics["density"] = static_cast<double>(per_row) / cols;
    r.metrics["validated_elements"] = static_cast<double>(y_count);
    r.metrics["validation_atol"] = static_cast<double>(atol);
    r.metrics["validation_rtol"] = static_cast<double>(rtol);
}

template<class T> struct sw_compute { using type = float; };
template<> struct sw_compute<double> { using type = double; };
template<class T> struct alignas(2 * sizeof(T)) sw_stored_complex { T x, y; };

template<class T> __host__ __device__ typename sw_compute<T>::type sw_load(T value) {
    if constexpr (std::is_same<T, __half>::value) return __half2float(value);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value) return __bfloat162float(value);
    else return value;
}

template<class T, class A> __host__ __device__ T sw_store(A value) {
    if constexpr (std::is_same<T, __half>::value) return __float2half_rn(static_cast<float>(value));
    else if constexpr (std::is_same<T, __nv_bfloat16>::value) return __float2bfloat16_rn(static_cast<float>(value));
    else return static_cast<T>(value);
}

template<class T> sw_stored_complex<T> sw_hermitian(std::size_t index, int nx, int ny, int nz,
                                                   unsigned salt, float scale) {
    const auto partner = swm::conjugate_index(index, nx, ny, nz);
    const auto canonical = std::min(index, partner);
    return {sw_store<T>(swm::input_value(canonical, salt) * scale),
            sw_store<T>(index == partner ? 0.0f : swm::input_value(canonical, salt + 3) * scale * (index < partner ? 1.0f : -1.0f))};
}

enum class sw_fft_operation { c2c, r2c, c2r };

template<class T> __global__ void sw_fft_unpack(const unsigned char* input,
        swm::complex_value<typename sw_compute<T>::type>* output, std::size_t count,
        std::size_t volume, std::size_t packed_volume, int nx, int ny, int nz, sw_fft_operation op) {
    using A = typename sw_compute<T>::type;
    const auto* complex_input = reinterpret_cast<const sw_stored_complex<T>*>(input);
    const auto* real_input = reinterpret_cast<const T*>(input);
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += std::size_t(blockDim.x) * gridDim.x) {
        if (op == sw_fft_operation::r2c) output[index] = {sw_load(real_input[index]), A(0)};
        else if (op == sw_fft_operation::c2c) output[index] = {sw_load(complex_input[index].x), sw_load(complex_input[index].y)};
        else {
            const auto source = swm::locate_hermitian(index, volume, packed_volume, nx, ny, nz);
            const auto value = complex_input[source.index];
            output[index] = {sw_load(value.x), source.conjugate ? -sw_load(value.y) : sw_load(value.y)};
        }
    }
}

template<class A> __global__ void sw_fft_reverse(const swm::complex_value<A>* input,
        swm::complex_value<A>* output, std::size_t count, int length, std::size_t stride) {
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += std::size_t(blockDim.x) * gridDim.x)
        output[index] = input[swm::bit_reversed_axis_index(index, length, stride)];
}

template<class A> __global__ void sw_fft_butterflies(swm::complex_value<A>* data,
        const swm::complex_value<A>* roots, std::size_t pair_count, int length,
        std::size_t stride, int half_span) {
    for (std::size_t pair = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pair_count; pair += std::size_t(blockDim.x) * gridDim.x) {
        const auto at = swm::locate_butterfly(pair, length, stride, half_span);
        swm::complex_value<A> low, high;
        swm::butterfly(data[at.low], data[at.high], roots[at.root], low, high);
        data[at.low] = low; data[at.high] = high;
    }
}

template<class A> __global__ void sw_fft_direct_axis(const swm::complex_value<A>* input,
        swm::complex_value<A>* output, const swm::complex_value<A>* roots,
        std::size_t count, int length, std::size_t stride) {
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += std::size_t(blockDim.x) * gridDim.x) {
        output[index] = swm::direct_axis_value(input, roots, index, length, stride);
    }
}

template<class T> __global__ void sw_fft_pack(const swm::complex_value<typename sw_compute<T>::type>* input,
        unsigned char* output, std::size_t count, std::size_t volume,
        std::size_t packed_volume, int nx, sw_fft_operation op) {
    auto* complex_output = reinterpret_cast<sw_stored_complex<T>*>(output);
    auto* real_output = reinterpret_cast<T*>(output);
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += std::size_t(blockDim.x) * gridDim.x) {
        if (op == sw_fft_operation::c2r) real_output[index] = sw_store<T>(input[index].x);
        else {
            const auto source = op == sw_fft_operation::r2c ?
                index / packed_volume * volume + swm::compact_to_full(index % packed_volume, nx) : index;
            complex_output[index] = {sw_store<T>(input[source].x), sw_store<T>(input[source].y)};
        }
    }
}

template<class T> void sw_fft(context& ctx, result& r, const std::string& op) {
    using A = typename sw_compute<T>::type;
    using C = sw_stored_complex<T>;
    using W = swm::complex_value<A>;
    static_assert(sizeof(C) == 2 * sizeof(T), "stored FFT complex values must be interleaved");
    constexpr bool narrow = std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value;
    const auto start = std::chrono::steady_clock::now();
    const auto requested_rank = ctx.opts.integer("rank", 1);
    if (requested_rank < 1 || requested_rank > 3) throw std::invalid_argument("FFT rank must be 1..3");
    const int rank = static_cast<int>(requested_rank);
    const auto requested_nx = sw_positive(ctx.opts, "nx", ctx.opts.integer("n", 4096));
    const auto requested_ny = sw_positive(ctx.opts, "ny", 16), requested_nz = sw_positive(ctx.opts, "nz", 16);
    const auto batch = sw_positive(ctx.opts, "batch", 1);
    if (requested_nx > INT32_MAX || requested_ny > INT32_MAX || requested_nz > INT32_MAX || batch > INT32_MAX)
        throw std::invalid_argument("software FFT uses the same int32 dimensions/batch contract as the cuFFT baseline");
    const int nx = static_cast<int>(requested_nx), ny = rank >= 2 ? static_cast<int>(requested_ny) : 1;
    const int nz = rank == 3 ? static_cast<int>(requested_nz) : 1;
    const auto volume = sw_product(sw_product(nx, ny), nz);
    if (volume > INT32_MAX) throw std::invalid_argument("FFT transform volume exceeds the shared int32 distance contract");
    const auto packed_volume = volume / nx * (nx / 2 + 1);
    const auto real_count = sw_product(volume, batch), packed_count = sw_product(packed_volume, batch);
    const auto complex_count = op == "c2c" ? real_count : packed_count;
    const auto input_bytes = op == "r2c" ? sw_product(real_count, sizeof(T)) : sw_product(complex_count, sizeof(C));
    const auto output_bytes = op == "c2r" ? sw_product(real_count, sizeof(T)) : sw_product(complex_count, sizeof(C));
    const auto direction_name = ctx.opts.get("direction", op == "c2r" ? "inverse" : "forward");
    if (direction_name != "forward" && direction_name != "inverse") throw std::invalid_argument("FFT direction must be forward or inverse");
    if ((op == "r2c" && direction_name != "forward") || (op == "c2r" && direction_name != "inverse")) {
        skip(r, "r2c requires forward direction and c2r requires inverse direction"); return;
    }
    const int sign = direction_name == "inverse" ? 1 : -1;
    const auto operation = op == "c2c" ? sw_fft_operation::c2c : op == "r2c" ? sw_fft_operation::r2c : sw_fft_operation::c2r;
    std::vector<int> lengths{nx};
    if (rank >= 2) lengths.push_back(ny);
    if (rank == 3) lengths.push_back(nz);
    std::vector<std::size_t> root_offsets;
    std::size_t root_count = 0;
    for (int length : lengths) { root_offsets.push_back(root_count); root_count = sw_sum(root_count, length); }
    const auto workspace_bytes = sw_sum(sw_product(sw_product(real_count, sizeof(W)), 2), sw_product(root_count, sizeof(W)));
    ctx.memory_guard(sw_sum(sw_sum(input_bytes, output_bytes), workspace_bytes));
    const float input_scale = narrow ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(volume))) : 1.0f;
    std::vector<T> real_input;
    std::vector<C> complex_input;
    if (op == "r2c") {
        real_input.resize(real_count);
        for (std::size_t i = 0; i < real_count; ++i) real_input[i] = sw_store<T>(swm::input_value(i, 11) * input_scale);
    } else {
        complex_input.resize(complex_count);
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t i = 0; i < (op == "c2c" ? volume : packed_volume); ++i) {
                auto& value = complex_input[b * (op == "c2c" ? volume : packed_volume) + i];
                if (op == "c2c") value = {sw_store<T>(swm::input_value(b * volume + i, 5) * input_scale),
                                          sw_store<T>(swm::input_value(b * volume + i, 19) * input_scale)};
                else value = sw_hermitian<T>(swm::compact_to_full(i, nx), nx, ny, nz, static_cast<unsigned>(b + 5), input_scale);
            }
        }
    }
    const long double pi = std::acos(-1.0L);
    std::vector<W> roots(root_count);
    for (std::size_t axis = 0; axis < lengths.size(); ++axis) {
        for (int index = 0; index < lengths[axis]; ++index) {
            const long double angle = sign * 2 * pi * index / lengths[axis];
            roots[root_offsets[axis] + index] = {static_cast<A>(std::cos(angle)), static_cast<A>(std::sin(angle))};
        }
    }
    device_buffer<unsigned char> input(input_bytes), output(output_bytes);
    device_buffer<W> first(real_count), second(real_count), device_roots(root_count);
    if (op == "r2c") sw_upload(ctx, reinterpret_cast<T*>(input.get()), real_input);
    else sw_upload(ctx, reinterpret_cast<C*>(input.get()), complex_input);
    sw_upload(ctx, device_roots.get(), roots);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    r.metrics["algorithm_setup_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const auto output_count = op == "c2r" ? real_count : complex_count;
    const int threads = sw_threads(ctx), blocks = sw_blocks(ctx, real_count, threads);
    auto launch = [&] {
        W* source = first.get(); W* destination = second.get();
        sw_fft_unpack<T><<<blocks, threads, 0, ctx.stream>>>(input.get(), source, real_count,
            volume, packed_volume, nx, ny, nz, operation);
        std::size_t stride = 1;
        for (std::size_t axis = 0; axis < lengths.size(); ++axis) {
            const int length = lengths[axis];
            const auto* axis_roots = device_roots.get() + root_offsets[axis];
            if (length > 1 && swm::power_of_two(length)) {
                sw_fft_reverse<<<blocks, threads, 0, ctx.stream>>>(source, destination, real_count, length, stride);
                std::swap(source, destination);
                for (int half_span = 1; half_span < length; half_span *= 2)
                    sw_fft_butterflies<<<blocks, threads, 0, ctx.stream>>>(source, axis_roots, real_count / 2, length, stride, half_span);
            } else if (length > 1) {
                sw_fft_direct_axis<<<blocks, threads, 0, ctx.stream>>>(source, destination, axis_roots, real_count, length, stride);
                std::swap(source, destination);
            }
            stride *= length;
        }
        sw_fft_pack<T><<<blocks, threads, 0, ctx.stream>>>(source, output.get(), output_count, volume, packed_volume, nx, operation);
    };
    r.samples_ms = ctx.measure(launch);
    std::vector<T> real_output;
    std::vector<C> complex_output;
    if (op == "c2r") real_output = sw_download(ctx, reinterpret_cast<T*>(output.get()), real_count);
    else complex_output = sw_download(ctx, reinterpret_cast<C*>(output.get()), complex_count);
    const auto samples = std::min(output_count, sw_positive(ctx.opts, "check-bins", 32));
    const long double atol = narrow ? (std::is_same<T, __half>::value ? 0.004L : 0.025L) *
        std::max(1.0L, std::sqrt(std::log2(static_cast<long double>(volume)))) : std::is_same<T, double>::value ?
        1e-10L * std::sqrt(static_cast<long double>(volume)) : 3e-5L * std::sqrt(static_cast<long double>(volume));
    const long double rtol = narrow ? (std::is_same<T, __half>::value ? 0.01L : 0.06L) :
        std::is_same<T, double>::value ? 1e-10L : 2e-4L;
    r.validation = "passed";
    for (std::size_t sample = 0; sample < samples; ++sample) {
        const auto index = samples == 1 ? 0 : sample * (output_count - 1) / (samples - 1);
        const auto out_volume = op == "r2c" ? packed_volume : volume;
        const auto b = index / out_volume, compact = index % out_volume;
        const auto full = op == "r2c" ? swm::compact_to_full(compact, nx) : compact;
        const auto fx = full % nx, fy = full / nx % ny, fz = full / (std::size_t(nx) * ny);
        std::complex<long double> reference(0, 0);
        for (std::size_t j = 0; j < volume; ++j) {
            const auto x = j % nx, y = j / nx % ny, z = j / (std::size_t(nx) * ny);
            const long double turns = static_cast<long double>(fx) * x / nx + static_cast<long double>(fy) * y / ny + static_cast<long double>(fz) * z / nz;
            const long double angle = sign * 2 * pi * turns;
            std::complex<long double> value;
            if (op == "r2c") value = static_cast<long double>(sw_load(real_input[b * volume + j]));
            else if (op == "c2c") value = {static_cast<long double>(sw_load(complex_input[b * volume + j].x)),
                                           static_cast<long double>(sw_load(complex_input[b * volume + j].y))};
            else {
                const auto generated = sw_hermitian<T>(j, nx, ny, nz, static_cast<unsigned>(b + 5), input_scale);
                value = {static_cast<long double>(sw_load(generated.x)), static_cast<long double>(sw_load(generated.y))};
            }
            reference += value * std::complex<long double>(std::cos(angle), std::sin(angle));
        }
        if (op == "c2r") sw_check(r, sw_load(real_output[index]), reference.real(), atol, rtol, index);
        else {
            sw_check(r, sw_load(complex_output[index].x), reference.real(), atol, rtol, index);
            sw_check(r, sw_load(complex_output[index].y), reference.imag(), atol, rtol, index);
        }
    }
    int direct_axes = 0, radix2_axes = 0;
    double direct_terms = 0, algorithm_flops = 0;
    for (int length : lengths) {
        if (length < 2) continue;
        if (swm::power_of_two(length)) { ++radix2_axes; algorithm_flops += 5.0 * real_count * std::log2(double(length)); }
        else { ++direct_axes; direct_terms += double(real_count) * length; algorithm_flops += 8.0 * real_count * length; }
    }
    r.execution = direct_axes ? "software_cuda_axis_fft_dft" : "software_cuda_radix2_fft";
    r.unit = "Gitem/s";
    r.work_per_iteration = static_cast<double>(real_count);
    r.bytes_per_iteration = static_cast<double>(input_bytes + output_bytes);
    r.metadata["work_domain"] = "item";
    r.metadata["library"] = "none";
    r.metadata["algorithm"] = "separable axis transforms; radix-2 bit-reversal/butterfly stages on power-of-two axes, direct DFT on other axes";
    r.metadata["compute_precision"] = std::is_same<T, double>::value ? "FP64 complex arithmetic" : "FP32 complex arithmetic; narrow inputs decoded before stages and outputs rounded once after all axes";
    r.metadata["input_contract"] = "same stored input values, compact Hermitian layout, input scaling and batch layout as the cuFFT baseline";
    r.metadata["normalization"] = "unnormalized forward and inverse, matching cuFFT; inverse is not divided by volume";
    r.metadata["timing_scope"] = "GPU unpack/Hermitian reconstruction, all axis kernels, and output conversion/packing; roots/input setup and CPU validation excluded";
    r.metadata["bandwidth_scope"] = "logical stored input+output bytes only; repeated full-complex workspace passes are not included and physical bandwidth is not inferred";
    r.metadata["c2r_input_policy"] = "immutable compact input; GPU reconstruction each launch, no separate restoration copy";
    r.metadata["validation_scope"] = samples == output_count ? "all output elements against independent long-double DFT" : "evenly spaced output elements against independent long-double DFT";
    r.metadata["operation_count"] = "input transform points; FFT/DFT arithmetic estimates are separate models, not measured instruction counts";
    r.metrics["temporary_storage_bytes"] = static_cast<double>(workspace_bytes);
    r.metrics["transform_points"] = static_cast<double>(volume);
    r.metrics["validated_elements"] = static_cast<double>(samples);
    r.metrics["validation_atol"] = static_cast<double>(atol);
    r.metrics["validation_rtol"] = static_cast<double>(rtol);
    r.metrics["input_scale"] = input_scale;
    r.metrics["radix2_axes"] = radix2_axes;
    r.metrics["direct_dft_axes"] = direct_axes;
    r.metrics["direct_dft_complex_terms"] = direct_terms;
    r.metrics["estimated_fft_flops"] = (op == "c2c" ? 5.0 : 2.5) * real_count * std::log2(double(volume));
    r.metrics["software_algorithm_arithmetic_model"] = algorithm_flops;
}

}  // namespace

bool run_systems_software(context& ctx, result& r) {
    const auto backend = ctx.opts.get("backend");
    if (backend != "software" && backend != "emulated") return false;
    const auto family = ctx.opts.get("family");
    if (family == "sparse") {
        const auto op = ctx.opts.get("op", "csr_spmv"), dtype = ctx.opts.get("dtype", "f32");
        if (op != "csr_spmv" && op != "csr_spmm") { skip(r, "software CSR operations are csr_spmv and csr_spmm"); return true; }
        if (dtype == "f32") sw_sparse<float>(ctx, r, op);
        else if (dtype == "f64") sw_sparse<double>(ctx, r, op);
        else skip(r, "software CSR storage types are f32 and f64");
        return true;
    }
    if (family == "fft") {
        const auto op = ctx.opts.get("op", "c2c"), dtype = ctx.opts.get("dtype", "c64");
        if (op != "c2c" && op != "r2c" && op != "c2r") { skip(r, "software FFT operations are c2c, r2c and c2r"); return true; }
        if (dtype == "f32" || dtype == "c64") sw_fft<float>(ctx, r, op);
        else if (dtype == "f64" || dtype == "c128") sw_fft<double>(ctx, r, op);
        else if (dtype == "f16") sw_fft<__half>(ctx, r, op);
        else if (dtype == "bf16") sw_fft<__nv_bfloat16>(ctx, r, op);
        else skip(r, "software FFT storage types are f16, bf16, f32/c64 and f64/c128");
        return true;
    }
    return false;
}

}  // namespace archbench
