// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
// Arithmetic/layout helpers shared by the real CUDA SIMT kernel and host tests.
// Including this header in a host test does not compile or validate that kernel.
#include "archbench/codec.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace archbench { namespace gemm_software {

enum class storage_kind { f16, bf16, f32, f64, c64, c128, i8, i32, e4m3, e5m2, e2m1 };
enum class input_rounding { unchanged, tf32, f16, bf16 };

ARCHBENCH_HD inline float rank1_input(bool operand_a,int row,int column) {
    return float(operand_a ? row%13-6 : column%11-5)/8.0f;
}
ARCHBENCH_HD inline double rank1_expected(int row,int column,int k) {
    return double(k)*double(rank1_input(true,row,0))*double(rank1_input(false,0,column));
}

ARCHBENCH_HD inline std::size_t matrix_index(int row, int column, int batch,
                                            int leading, std::size_t stride,
                                            bool transposed) {
    return std::size_t(batch) * stride +
        (transposed ? std::size_t(row) * leading + column
                    : std::size_t(column) * leading + row);
}

// The scale tensor is a tiled hardware layout, not a flat row-major array.
ARCHBENCH_HD inline std::size_t scale_offset(int outer, int reduction, int k, int vector_size) {
    const std::size_t inner = reduction / vector_size;
    const std::size_t tile_columns = ((std::size_t(k) + vector_size - 1) / vector_size + 3) / 4;
    return ((std::size_t(outer) / 128) * tile_columns + inner / 4) * 512 +
           (outer % 32) * 16 + ((outer % 128) / 32) * 4 + inner % 4;
}

ARCHBENCH_HD inline float round_mantissa_rne(float x, unsigned removed_bits) {
    auto bits = codec::float_bits(x);
    if ((bits & 0x7f800000u) == 0x7f800000u || removed_bits == 0) return x;
    const auto mask = (1u << removed_bits) - 1;
    const auto remainder = bits & mask, halfway = 1u << (removed_bits - 1);
    bits &= ~mask;
    if (remainder > halfway || (remainder == halfway && ((bits >> removed_bits) & 1u)))
        bits += 1u << removed_bits;
    return codec::from_bits(bits);
}

// IEEE binary16 conversion, including subnormal ties, infinities and NaNs.
// codec::encode deliberately saturates finite; it is not used for this policy.
ARCHBENCH_HD inline std::uint16_t half_bits_rne(float x) {
    const auto bits = codec::float_bits(x);
    const unsigned sign = (bits >> 16) & 0x8000u;
    const unsigned source_exponent = (bits >> 23) & 255u;
    unsigned mantissa = bits & 0x7fffffu;
    if (source_exponent == 255u)
        return std::uint16_t(sign | (mantissa ? 0x7e00u : 0x7c00u));
    const int exponent = int(source_exponent) - 127 + 15;
    if (exponent >= 31) return std::uint16_t(sign | 0x7c00u);
    if (exponent < -10) return std::uint16_t(sign);
    if (exponent <= 0) {
        mantissa |= 0x800000u;
        const unsigned shift = unsigned(14 - exponent);
        unsigned value = mantissa >> shift;
        const unsigned discarded = mantissa & ((1u << shift) - 1);
        const unsigned half = 1u << (shift - 1);
        value += discarded > half || (discarded == half && (value & 1u));
        return std::uint16_t(sign | value);
    }
    unsigned value = (unsigned(exponent) << 10) | (mantissa >> 13);
    const unsigned discarded = mantissa & 8191u;
    value += discarded > 4096u || (discarded == 4096u && (value & 1u));
    return std::uint16_t(sign | value);
}

ARCHBENCH_HD inline float round_input(float x, input_rounding mode) {
    if (mode == input_rounding::tf32) return round_mantissa_rne(x, 13);
    if (mode == input_rounding::bf16) return round_mantissa_rne(x, 16);
    if (mode == input_rounding::f16)
        return codec::decode(half_bits_rne(x),
            {codec::kind::binary_float, 16, 5, 10, 15, codec::special::ieee});
    return x;
}

template<class T> ARCHBENCH_HD T load_component(const void* buffer, storage_kind type,
                                               std::size_t index, bool imaginary = false) {
    if (imaginary && type != storage_kind::c64 && type != storage_kind::c128) return T(0);
    const auto* bytes = static_cast<const std::uint8_t*>(buffer);
    switch (type) {
    case storage_kind::f32: return T(static_cast<const float*>(buffer)[index]);
    case storage_kind::f64: return T(static_cast<const double*>(buffer)[index]);
    case storage_kind::c64: return T(static_cast<const float*>(buffer)[2 * index + imaginary]);
    case storage_kind::c128: return T(static_cast<const double*>(buffer)[2 * index + imaginary]);
    case storage_kind::i8: return T(static_cast<const std::int8_t*>(buffer)[index]);
    case storage_kind::i32: return T(static_cast<const std::int32_t*>(buffer)[index]);
    case storage_kind::f16: return T(codec::decode(
        unsigned(bytes[2 * index]) | (unsigned(bytes[2 * index + 1]) << 8),
        {codec::kind::binary_float, 16, 5, 10, 15, codec::special::ieee}));
    case storage_kind::bf16: return T(codec::from_bits(
        (unsigned(bytes[2 * index]) | (unsigned(bytes[2 * index + 1]) << 8)) << 16));
    case storage_kind::e4m3: return T(codec::decode(bytes[index],
        {codec::kind::binary_float, 8, 4, 3, 7, codec::special::terminal_nan}));
    case storage_kind::e5m2: return T(codec::decode(bytes[index],
        {codec::kind::binary_float, 8, 5, 2, 15, codec::special::ieee}));
    case storage_kind::e2m1: return T(codec::decode((bytes[index / 2] >> ((index & 1) * 4)) & 15u,
        {codec::kind::binary_float, 4, 2, 1, 1, codec::special::finite_only}));
    }
    return T(0);
}

ARCHBENCH_HD inline float load_scale(const std::uint8_t* data, int outer, int reduction,
                                     int k, int vector_size) {
    if (vector_size == 0) return codec::load_scale(data, 0, codec::scale_kind::f32);
    return codec::load_scale(data, scale_offset(outer,reduction,k,vector_size),
        vector_size == 16 ? codec::scale_kind::e4m3 : codec::scale_kind::e8m0);
}

// Candidates may reach this selector only after the independently sampled
// output check. Reject nonfinite/zero timings even if an API call succeeded.
struct candidate_score {
    int index = -1;
    bool validated = false;
    std::vector<double> samples_ms;
};
inline double candidate_median(const candidate_score& candidate) {
    if (!candidate.validated || candidate.samples_ms.empty())
        return std::numeric_limits<double>::infinity();
    auto samples = candidate.samples_ms;
    for (double x : samples)
        if (!std::isfinite(x) || x <= 0) return std::numeric_limits<double>::infinity();
    std::sort(samples.begin(),samples.end());
    const auto n = samples.size();
    return n & 1 ? samples[n / 2] : .5 * samples[n / 2 - 1] + .5 * samples[n / 2];
}
inline int fastest_validated(const std::vector<candidate_score>& candidates) {
    int selected = -1;
    double best = std::numeric_limits<double>::infinity();
    for (const auto& candidate : candidates) {
        const double median = candidate_median(candidate);
        if (median < best || (median == best && std::isfinite(median) &&
                             (selected < 0 || candidate.index < selected))) {
            best = median; selected = candidate.index;
        }
    }
    return selected;
}

}} // namespace archbench::gemm_software
