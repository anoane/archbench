// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Pure host accounting shared with executable host regression tests. These
// counts describe requested mathematical work, never sampled SASS counters.
namespace archbench::tensor_model {

inline int reset_interval(int64_t requested) {
    if (requested < 0 || requested > 1000000)
        throw std::invalid_argument("reset-every must be 0 (disabled) or 1..1000000 MMA iterations");
    return static_cast<int>(requested);
}

inline int final_segment_iterations(int iterations, int reset_every) {
    if (iterations < 1 || reset_every < 0) throw std::invalid_argument("invalid tensor segment");
    return reset_every ? (iterations - 1) % reset_every + 1 : iterations;
}

inline int largest_segment_iterations(int iterations, int reset_every) {
    if (iterations < 1 || reset_every < 0) throw std::invalid_argument("invalid tensor segment");
    return reset_every ? std::min(iterations, reset_every) : iterations;
}

inline uint64_t reset_events(int iterations, int reset_every) {
    if (iterations < 1 || reset_every < 0) throw std::invalid_argument("invalid tensor segment");
    return reset_every ? uint64_t(iterations - 1) / uint64_t(reset_every) : 0;
}

inline uint64_t multiply(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::overflow_error("tensor work count exceeds uint64_t");
    return a * b;
}

struct work_counts {
    uint64_t warps;
    uint64_t logical_tiles;
    uint64_t operations_per_tile;
    uint64_t operations;
};

inline work_counts count_work(uint64_t lanes, int ilp, int iterations,
                              int m, int n, int k, bool binary) {
    if (!lanes || lanes % 32 || ilp < 1 || iterations < 1 || m < 1 || n < 1 || k < 1)
        throw std::invalid_argument("invalid tensor work dimensions");
    const uint64_t warps = lanes / 32;
    const auto tiles = multiply(multiply(warps, uint64_t(ilp)), uint64_t(iterations));
    const auto tile_work = multiply(multiply(multiply(uint64_t(m), uint64_t(n)), uint64_t(k)), binary ? 1 : 2);
    return {warps, tiles, tile_work, multiply(tiles, tile_work)};
}

}  // namespace archbench::tensor_model
