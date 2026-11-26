// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
// Portable numeric encodings used by the emulation baselines, not vendor file layouts.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#ifdef __CUDACC__
#define ARCHBENCH_HD __host__ __device__
#else
#define ARCHBENCH_HD
#endif

namespace archbench {
namespace codec {

enum class kind : int { signed_integer, unsigned_integer, binary_float, normal_float };
enum class special : int { finite_only, ieee, terminal_nan };
struct format {
  kind type = kind::binary_float;
  int bits = 4;
  int exponent_bits = 2;
  int mantissa_bits = 1;
  int bias = 1;
  special specials = special::finite_only;
};

inline format parse(const std::string& name) {
  if (name.size() >= 2 && (name[0] == 's' || name[0] == 'u')) {
    std::size_t consumed = 0;
    int bits = std::stoi(name.substr(1), &consumed);
    if (consumed != name.size() - 1 || bits < 1 || bits > 16)
      throw std::invalid_argument("quant integer format must be s1..s16 or u1..u16");
    return {name[0] == 's' ? kind::signed_integer : kind::unsigned_integer, bits, 0, 0, 0, special::finite_only};
  }
  if (name == "e2m1") return {kind::binary_float, 4, 2, 1, 1, special::finite_only};
  if (name == "e2m3") return {kind::binary_float, 6, 2, 3, 1, special::finite_only};
  if (name == "e3m2") return {kind::binary_float, 6, 3, 2, 3, special::finite_only};
  if (name == "e4m3") return {kind::binary_float, 8, 4, 3, 7, special::terminal_nan};
  if (name == "e5m2") return {kind::binary_float, 8, 5, 2, 15, special::ieee};
  if (name == "f16") return {kind::binary_float, 16, 5, 10, 15, special::ieee};
  if (name == "bf16") return {kind::binary_float, 16, 8, 7, 127, special::ieee};
  if (name == "nf4") return {kind::normal_float, 4, 0, 0, 0, special::finite_only};
  throw std::invalid_argument("unknown numeric format: " + name);
}

ARCHBENCH_HD inline std::uint32_t float_bits(float value) {
#ifdef __CUDA_ARCH__
  return __float_as_uint(value);
#else
  std::uint32_t result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
#endif
}
ARCHBENCH_HD inline float from_bits(std::uint32_t value) {
#ifdef __CUDA_ARCH__
  return __uint_as_float(value);
#else
  float result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
#endif
}

// NF4 codebook values are numeric format constants from bitsandbytes, MIT licensed.
// See THIRD_PARTY_NOTICES.md. Packing here is low-bit first, not bitsandbytes layout.
ARCHBENCH_HD inline float normal_value(unsigned code) {
  switch (code & 15u) {
    case 0: return -1.0f;
    case 1: return -0.6961928009986877f;
    case 2: return -0.5250730514526367f;
    case 3: return -0.39491748809814453f;
    case 4: return -0.28444138169288635f;
    case 5: return -0.18477343022823334f;
    case 6: return -0.09105003625154495f;
    case 7: return 0.0f;
    case 8: return 0.07958029955625534f;
    case 9: return 0.16093020141124725f;
    case 10: return 0.24611230194568634f;
    case 11: return 0.33791524171829224f;
    case 12: return 0.44070982933044434f;
    case 13: return 0.5626170039176941f;
    case 14: return 0.7229568362236023f;
    default: return 1.0f;
  }
}

ARCHBENCH_HD inline std::uint32_t positive_max_code(format f) {
  const unsigned exp_max = (1u << f.exponent_bits) - 1;
  const unsigned mant_max = (1u << f.mantissa_bits) - 1;
  if (f.specials == special::ieee) return ((exp_max - 1) << f.mantissa_bits) | mant_max;
  if (f.specials == special::terminal_nan) return (exp_max << f.mantissa_bits) | (mant_max - 1);
  return (exp_max << f.mantissa_bits) | mant_max;
}

ARCHBENCH_HD inline float decode(std::uint32_t code, format f) {
  const unsigned mask = (1u << f.bits) - 1;
  code &= mask;
  if (f.type == kind::normal_float) return normal_value(code);
  if (f.type == kind::unsigned_integer) return static_cast<float>(code);
  if (f.type == kind::signed_integer) {
    const int value = (code & (1u << (f.bits - 1))) ? int(code) - int(1u << f.bits) : int(code);
    return static_cast<float>(value);
  }
  const bool negative = code >> (f.bits - 1);
  const unsigned mant = code & ((1u << f.mantissa_bits) - 1);
  const unsigned exp = (code >> f.mantissa_bits) & ((1u << f.exponent_bits) - 1);
  const unsigned max_exp = (1u << f.exponent_bits) - 1;
  if (f.specials == special::ieee && exp == max_exp)
    return from_bits((negative ? 0x80000000u : 0u) | (mant ? 0x7fc00000u : 0x7f800000u));
  if (f.specials == special::terminal_nan && exp == max_exp && mant == ((1u << f.mantissa_bits) - 1))
    return from_bits(0x7fc00000u);
  float value = exp == 0 ? ldexpf(static_cast<float>(mant), 1 - f.bias - f.mantissa_bits)
                        : ldexpf(static_cast<float>((1u << f.mantissa_bits) + mant), int(exp) - f.bias - f.mantissa_bits);
  return negative ? -value : value;
}

ARCHBENCH_HD inline float round_even(float value) {
  const float base = floorf(value);
  const float fraction = value - base;
  return base + ((fraction > 0.5f || (fraction == 0.5f && fmodf(base, 2.0f) != 0.0f)) ? 1.0f : 0.0f);
}

ARCHBENCH_HD inline std::uint32_t encode(float value, format f) {
  const bool nan = (float_bits(value) & 0x7fffffffu) > 0x7f800000u;
  if (nan) {
    if (f.type == kind::binary_float && f.specials != special::finite_only)
      return ((1u << f.exponent_bits) - 1u) << f.mantissa_bits | ((1u << f.mantissa_bits) - 1u);
    return f.type == kind::normal_float ? 7 : 0;
  }
  if (f.type == kind::signed_integer || f.type == kind::unsigned_integer) {
    const float lo = f.type == kind::signed_integer ? -float(1u << (f.bits - 1)) : 0.0f;
    const float hi = f.type == kind::signed_integer ? float((1u << (f.bits - 1)) - 1) : float((1u << f.bits) - 1);
    const float clipped = fminf(hi, fmaxf(lo, value));
    return std::uint32_t(int(round_even(clipped))) & ((1u << f.bits) - 1);
  }
  if (f.type == kind::normal_float) {
    value = fminf(1.0f, fmaxf(-1.0f, value));
    unsigned best = 0;
    float distance = fabsf(value - normal_value(0));
    for (unsigned code = 1; code < 16; ++code) {
      const float next = fabsf(value - normal_value(code));
      if (next < distance || (next == distance && (code & 1u) == 0 && (best & 1u))) {
        best = code; distance = next;
      }
    }
    return best;
  }
  const unsigned sign = (float_bits(value) >> 31) << (f.bits - 1);
  value = fabsf(value);
  unsigned lo = 0, hi = positive_max_code(f);
  if (value >= decode(hi, f)) return sign | hi;  // saturate finite, including infinities
  // Monotonic positive codes let us avoid a format-specific rounding shortcut.
  while (lo + 1 < hi) {
    const unsigned mid = (lo + hi) / 2;
    if (decode(mid, f) <= value) lo = mid;
    else hi = mid;
  }
  const double dlo = double(value) - double(decode(lo, f));
  const double dhi = double(decode(hi, f)) - double(value);
  const unsigned chosen = dlo < dhi ? lo : (dhi < dlo ? hi : ((lo & 1u) ? hi : lo));
  return sign | chosen;
}

ARCHBENCH_HD inline float maximum(format f) {
  if (f.type == kind::normal_float) return 1;
  if (f.type == kind::signed_integer) return float((1u << (f.bits - 1)) - 1u);
  if (f.type == kind::unsigned_integer) return float((1u << f.bits) - 1u);
  return decode(positive_max_code(f), f);
}

ARCHBENCH_HD inline std::size_t packed_bytes(std::size_t count, int bits) { return (count * bits + 7) / 8; }

// Bounds: buffer must hold ceil(count*bits/8), code width is 1..16.
ARCHBENCH_HD inline std::uint32_t unpack(const std::uint8_t* data, std::size_t index, int bits) {
  const std::size_t bit = index * bits;
  const std::size_t byte = bit / 8;
  const unsigned shift = unsigned(bit % 8);
  std::uint32_t value = data[byte];
  if (shift + bits > 8) value |= std::uint32_t(data[byte + 1]) << 8;
  if (shift + bits > 16) value |= std::uint32_t(data[byte + 2]) << 16;
  return (value >> shift) & ((1u << bits) - 1u);
}
ARCHBENCH_HD inline void pack(std::uint8_t* data, std::size_t index, int bits, std::uint32_t code) {
  // One owner writes a complete group; no concurrent read-modify-write races.
  const std::size_t bit = index * bits;
  for (int b = 0; b < bits; ++b) {
    const auto byte = (bit + b) / 8;
    const unsigned shift = unsigned((bit + b) % 8);
    data[byte] = std::uint8_t((data[byte] & ~(1u << shift)) | (((code >> b) & 1u) << shift));
  }
}

enum class scale_kind : int { f32, e8m0, e4m3 };
inline scale_kind parse_scale(const std::string& name) {
  if (name == "f32") return scale_kind::f32;
  if (name == "e8m0") return scale_kind::e8m0;
  if (name == "e4m3") return scale_kind::e4m3;
  throw std::invalid_argument("scale format must be f32, e8m0 or e4m3");
}
ARCHBENCH_HD inline int scale_bytes(scale_kind kind) { return kind == scale_kind::f32 ? 4 : 1; }
ARCHBENCH_HD inline unsigned scale_encode_e8m0(float scale) {
  if (!(scale > 0)) return 0;
  int exp;
  const float mant = frexpf(scale, &exp);
  int encoded = (mant == 0.5f ? exp - 1 : exp) + 127;  // upward power-of-two
  return unsigned(encoded < 0 ? 0 : (encoded > 254 ? 254 : encoded));
}
ARCHBENCH_HD inline float load_scale(const std::uint8_t* data, std::size_t index, scale_kind kind) {
  if (kind == scale_kind::f32) {
    const auto* p = data + index * 4;
    return from_bits(std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24));
  }
  if (kind == scale_kind::e8m0) return data[index] == 255 ? from_bits(0x7fc00000u) : ldexpf(1.0f, int(data[index]) - 127);
  return decode(data[index], {kind::binary_float, 8, 4, 3, 7, special::terminal_nan});
}
ARCHBENCH_HD inline void store_scale(std::uint8_t* data, std::size_t index, scale_kind kind, float scale) {
  if (kind == scale_kind::f32) {
    const auto value = float_bits(scale);
    for (unsigned i = 0; i < 4; ++i) data[index * 4 + i] = std::uint8_t(value >> (i * 8));
  } else if (kind == scale_kind::e8m0) data[index] = std::uint8_t(scale_encode_e8m0(scale));
  else data[index] = std::uint8_t(encode(scale, {kind::binary_float, 8, 4, 3, 7, special::terminal_nan}));
}

ARCHBENCH_HD inline void encode_group(const float* input, std::size_t count, format f,
                         std::uint8_t* packed, std::uint8_t* scales, std::size_t group_index,
                         scale_kind scale_type, float* offsets) {
  float maximum_abs = 0, lowest = count ? input[0] : 0, highest = lowest;
  for (std::size_t i = 0; i < count; ++i) {
    maximum_abs = fmaxf(maximum_abs, fabsf(input[i]));
    lowest = fminf(lowest, input[i]); highest = fmaxf(highest, input[i]);
  }
  const bool affine = f.type == kind::unsigned_integer;
  float denominator = maximum(f);
  if (denominator < 1) denominator = 1;
  float scale = (affine ? highest - lowest : maximum_abs) / denominator;
  if (!(scale > 0) || (f.type == kind::binary_float && f.bits == 16)) scale = 1;
  store_scale(scales, group_index, scale_type, scale);
  scale = load_scale(scales, group_index, scale_type);
  // E4M3 can round a tiny scale to zero. Keep the smallest positive E4M3 value.
  if (!(scale > 0)) { store_scale(scales, group_index, scale_type, 0.001953125f); scale = load_scale(scales, group_index, scale_type); }
  const float offset = affine ? lowest : 0;
  offsets[group_index] = offset;
  for (std::size_t i = 0; i < packed_bytes(count, f.bits); ++i) packed[i] = 0;
  for (std::size_t i = 0; i < count; ++i) pack(packed, i, f.bits, encode((input[i] - offset) / scale, f));
}

}  // namespace codec
}  // namespace archbench
