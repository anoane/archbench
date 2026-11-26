#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Execute the CUDA two-limb algorithms as host C++, against native __int128.

This validates arithmetic semantics and masks for every width 1..128. It does
NOT compile CUDA, validate PTX syntax, or establish GPU performance/correctness.
The source extraction deliberately fails if the CUDA helper boundaries change.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import tempfile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]
    source = (root / "src" / "alu.cu").read_text()
    enum = re.search(r"enum class int_op \{[^}]+\};", source)
    if enum is None:
        raise RuntimeError("integer operation enum not found")
    begin = source.index("struct limb_integer {")
    end = source.index("#if defined(__SIZEOF_INT128__)", begin)
    helpers = source[begin:end]
    # The seed is input generation, unrelated to arithmetic under validation.
    seed_begin = helpers.index("__host__ __device__ limb_integer limb_seed(")
    seed_end = helpers.index("__device__ __forceinline__ limb_integer limb_mul(", seed_begin)
    helpers = helpers[:seed_begin] + helpers[seed_end:]
    helpers = re.sub(r"asm volatile\(\"mov\.b64[^;]+;\"[^;]+;", "", helpers)
    if "asm volatile" in helpers:
        raise RuntimeError("unexpected assembly in host-testable limb helper region")
    helpers = helpers.replace("__host__", "").replace("__device__", "").replace("__forceinline__", "inline")
    helpers = re.sub(r"^\s*#pragma unroll.*$", "", helpers, flags=re.MULTILINE)
    reference_begin = source.index("using host_uint128", end)
    reference_end = source.index("#endif", reference_begin)
    reference = source[reference_begin:reference_end]
    program = r'''
#include <cstdint>
#include <iostream>
#include <stdexcept>
static uint64_t __umul64hi(uint64_t a, uint64_t b) {
    return uint64_t((static_cast<unsigned __int128>(a) * b) >> 64);
}
'''
    program += enum.group(0) + "\n" + helpers + "\n" + reference
    program += r'''
static uint64_t rng(uint64_t& state) {
    state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
template<int_op Op>
static void check(limb_integer a, limb_integer b, limb_integer c, unsigned width, bool sign, size_t& checks) {
    const auto got = limb_device<Op>(a, b, c, width, sign);
    const auto want = limb_reference<Op>(a, b, c, width, sign);
    ++checks;
    if (got.lo != want.lo || got.hi != want.hi) {
        std::cerr << "Mismatch width=" << width << " signed=" << sign << " op=" << int(Op) << "\n";
        throw std::runtime_error("limb arithmetic mismatch");
    }
}
int main() {
    uint64_t state = 0x4a9c3148ff06e271ull;
    size_t checks = 0;
    for (unsigned width = 1; width <= 128; ++width) {
        for (bool sign : {false, true}) {
            for (int sample = 0; sample < 160; ++sample) {
                auto a = limb_mask({rng(state), rng(state)}, width);
                auto b = limb_mask({rng(state), rng(state)}, width);
                auto c = limb_mask({rng(state), rng(state)}, width);
                if (sample == 0) a = {0, 0};
                if (sample == 1) a = limb_mask({~uint64_t(0), ~uint64_t(0)}, width);
                if (sample == 2 || sample == 3) a = limb_shl({1, 0}, width - 1);
                if (sample == 3) b = limb_mask({~uint64_t(0), ~uint64_t(0)}, width);
                if (sample == 4) b = limb_shl({1, 0}, width - 1);
                if (sample == 5) b = {1, 0};
                if (sample == 6) b = limb_mask({~uint64_t(0), ~uint64_t(0)}, width);
                if (sample == 7) a = limb_sub(limb_shl({1, 0}, width - 1), {1, 0});
                if (limb_zero(b)) b = {1, 0};
                #define CHECK(OP) check<int_op::OP>(a,b,c,width,sign,checks)
                CHECK(add); CHECK(sub); CHECK(mul); CHECK(mad);
                CHECK(div); CHECK(rem); CHECK(min); CHECK(max);
                CHECK(bit_and); CHECK(bit_or); CHECK(bit_xor); CHECK(shl); CHECK(shr);
                #undef CHECK
            }
        }
    }
    std::cout << checks << " host-executed limb transitions passed for widths 1..128, both signs\n";
    std::cout << "CUDA compilation and GPU validation remain separate gates\n";
}
'''
    with tempfile.TemporaryDirectory(prefix="archbench-limb-") as folder:
        cpp = pathlib.Path(folder) / "limb_test.cpp"
        binary = pathlib.Path(folder) / "limb_test"
        cpp.write_text(program)
        subprocess.run([args.compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", str(cpp), "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
