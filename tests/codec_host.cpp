// SPDX-License-Identifier: AGPL-3.0-or-later
#include "archbench/codec.hpp"
#include <array>
#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
using namespace archbench::codec;

int main(int argc, char** argv) {
  if (argc >= 3 && std::string(argv[1]) == "decode") {
    const auto f = parse(argv[2]);
    for (unsigned code = 0; code < (1u << f.bits); ++code)
      std::cout << code << ' ' << std::hexfloat << decode(code, f) << std::defaultfloat << '\n';
    return 0;
  }
  if (argc >= 3 && std::string(argv[1]) == "encode") {
    const auto f = parse(argv[2]);
    std::string text;
    while (std::cin >> text) std::cout << encode(std::stof(text), f) << '\n';
    return 0;
  }
  unsigned long long checks = 0;
  for (const auto& name : {"e2m1", "e2m3", "e3m2", "e4m3", "e5m2", "f16", "bf16", "nf4"}) {
    const auto f = parse(name);
    for (unsigned code = 0; code < (1u << f.bits); ++code) {
      const auto value = decode(code, f);
      if (std::isfinite(value)) {
        const auto back = encode(value, f);
        if (back != code) { std::cerr << name << " roundtrip " << code << " -> " << back << '\n'; return 1; }
        ++checks;
      }
    }
  }
  for (int width = 1; width <= 16; ++width) {
    const std::size_t count = 131;
    std::vector<unsigned char> data(packed_bytes(count, width) + 8, 0xa5);
    const auto sentinel = data;
    for (std::size_t i = 0; i < count; ++i) pack(data.data(), i, width, unsigned(i * 1709) & ((1u << width) - 1));
    for (std::size_t i = 0; i < count; ++i) {
      assert(unpack(data.data(), i, width) == (unsigned(i * 1709) & ((1u << width) - 1)));
      ++checks;
    }
    for (auto i = packed_bytes(count, width); i < data.size(); ++i) assert(data[i] == sentinel[i]);
    for (char sign : {'s', 'u'}) {
      const auto f = parse(std::string(1, sign) + std::to_string(width));
      for (unsigned code = 0; code < (1u << width); ++code) {
        assert(encode(decode(code, f), f) == code); ++checks;
      }
    }
  }
  assert(decode(7, parse("e2m1")) == 6);
  assert(decode(31, parse("e2m3")) == 7.5f);
  assert(decode(31, parse("e3m2")) == 28);
  assert(decode(126, parse("e4m3")) == 448);
  assert(decode(123, parse("e5m2")) == 57344);
  assert(encode(1.25f, parse("e2m1")) == 2);
  assert(encode(1.75f, parse("e2m1")) == 4);
  assert(scale_encode_e8m0(1) == 127);
  assert(scale_encode_e8m0(1.1f) == 128);
  assert(scale_encode_e8m0(0.5f) == 126);
  std::cout << "codec checks: " << checks << " passed\n";
}
