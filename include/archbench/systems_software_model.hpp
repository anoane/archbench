// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define ARCHBENCH_SOFTWARE_HD __host__ __device__
#else
#define ARCHBENCH_SOFTWARE_HD
#endif

namespace archbench::systems_software_model {

// Matches the stored-input generator in the library-backed systems suite.
inline float input_value(std::size_t index, unsigned salt = 0) {
    const auto hash = static_cast<std::uint32_t>(index) * 1664525u +
                      1013904223u + salt * 2246822519u;
    return static_cast<float>(static_cast<int>((hash >> 16) % 257u) - 128) / 128.0f;
}

template<class T> struct complex_value { T x, y; };

template<class T>
ARCHBENCH_SOFTWARE_HD inline complex_value<T> multiply(complex_value<T> a, complex_value<T> b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

template<class T>
ARCHBENCH_SOFTWARE_HD inline void butterfly(complex_value<T> a, complex_value<T> b,
                                           complex_value<T> w,
                                           complex_value<T>& low, complex_value<T>& high) {
    const auto product = multiply(b, w);
    low = {a.x + product.x, a.y + product.y};
    high = {a.x - product.x, a.y - product.y};
}

ARCHBENCH_SOFTWARE_HD inline std::size_t conjugate_index(std::size_t index, int nx, int ny, int nz) {
    const auto x = index % nx, y = index / nx % ny, z = index / (std::size_t(nx) * ny);
    const auto cx = x ? nx - x : 0, cy = y ? ny - y : 0, cz = z ? nz - z : 0;
    return (cz * ny + cy) * nx + cx;
}

ARCHBENCH_SOFTWARE_HD inline std::size_t compact_to_full(std::size_t compact, int nx) {
    return compact / (nx / 2 + 1) * nx + compact % (nx / 2 + 1);
}

ARCHBENCH_SOFTWARE_HD inline std::size_t full_to_compact(std::size_t index, int nx) {
    return index / nx * (nx / 2 + 1) + index % nx;
}

ARCHBENCH_SOFTWARE_HD inline std::size_t bit_reversed_axis_index(std::size_t index,
                                                                int length, std::size_t stride) {
    unsigned coordinate = static_cast<unsigned>((index / stride) % length), reversed = 0;
    for (unsigned bound = static_cast<unsigned>(length); bound > 1; bound >>= 1) {
        reversed = (reversed << 1) | (coordinate & 1u);
        coordinate >>= 1;
    }
    return index - ((index / stride) % length) * stride + reversed * stride;
}

struct butterfly_location { std::size_t low, high; int root; };

ARCHBENCH_SOFTWARE_HD inline butterfly_location locate_butterfly(std::size_t pair, int length,
                                                                 std::size_t stride, int half_span) {
    const auto inner = pair % stride;
    const auto line = pair / stride;
    const auto axis_pair = line % (length / 2);
    const auto outer = line / (length / 2);
    const auto element = axis_pair % half_span;
    const auto low = outer * length * stride + (axis_pair / half_span * (2 * half_span) + element) * stride + inner;
    return {low, low + half_span * stride, static_cast<int>(element) * (length / (2 * half_span))};
}

ARCHBENCH_SOFTWARE_HD inline bool power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

template<class T>
ARCHBENCH_SOFTWARE_HD inline complex_value<T> direct_axis_value(const complex_value<T>* input,
        const complex_value<T>* roots, std::size_t index, int length, std::size_t stride) {
    const auto frequency = (index / stride) % length, base = index - frequency * stride;
    complex_value<T> value{T(0), T(0)};
    for (int sample = 0; sample < length; ++sample) {
        const auto product = multiply(input[base + std::size_t(sample) * stride],
                                      roots[(frequency * std::size_t(sample)) % length]);
        value.x += product.x; value.y += product.y;
    }
    return value;
}

struct hermitian_location { std::size_t index; bool conjugate; };
ARCHBENCH_SOFTWARE_HD inline hermitian_location locate_hermitian(std::size_t index,
        std::size_t volume, std::size_t packed_volume, int nx, int ny, int nz) {
    const auto batch = index / volume, full = index % volume;
    const bool conjugate = full % nx > static_cast<std::size_t>(nx / 2);
    const auto stored = conjugate ? conjugate_index(full, nx, ny, nz) : full;
    return {batch * packed_volume + full_to_compact(stored, nx), conjugate};
}

}  // namespace archbench::systems_software_model
#undef ARCHBENCH_SOFTWARE_HD
