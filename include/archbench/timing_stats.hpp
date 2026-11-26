// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace archbench {

// Host-only arithmetic shared by timing collection, serialization, and host tests.
struct timing_statistics {
  std::size_t count = 0;
  bool valid = false;
  double minimum = 0, maximum = 0, median = 0, mean = 0;
  double p10 = 0, p90 = 0, mad = 0;
  double stddev = std::numeric_limits<double>::quiet_NaN();
  double cv = std::numeric_limits<double>::quiet_NaN();
};

inline timing_statistics analyze_timing(const std::vector<double>& values) {
  timing_statistics out;
  out.count = values.size();
  if (values.empty()) return out;
  for (double v : values) if (!std::isfinite(v) || v <= 0) return out;
  std::vector<double> sorted(values);
  std::sort(sorted.begin(), sorted.end());
  auto quantile = [](const std::vector<double>& x, double p) {
    const double position = (x.size() - 1) * p;
    const auto lower = static_cast<std::size_t>(position);
    const double fraction = position - lower;
    return x[lower] * (1 - fraction) + x[std::min(lower + 1, x.size() - 1)] * fraction;
  };
  out.minimum = sorted.front(); out.maximum = sorted.back();
  out.median = quantile(sorted, .5); out.p10 = quantile(sorted, .1); out.p90 = quantile(sorted, .9);
  long double mean = 0, m2 = 0;
  std::size_t n = 0;
  for (double v : values) {
    const long double delta = v - mean;
    mean += delta / ++n;
    m2 += delta * (v - mean);
  }
  out.mean = static_cast<double>(mean);
  if (n > 1) {
    out.stddev = std::sqrt(static_cast<double>(std::max(0.L, m2 / (n - 1))));
    out.cv = out.stddev / out.mean;
  }
  for (auto& v : sorted) v = std::abs(v - out.median);
  std::sort(sorted.begin(), sorted.end());
  out.mad = quantile(sorted, .5);
  out.valid = true;
  return out;
}

struct timing_policy {
  int minimum_samples = 7, maximum_samples = 21, warmup = 3, max_repeats = 10000, repeats = 0;
  double target_ms = 20, target_cv = .05, warmup_ms = 0;
  bool needs_more(const std::vector<double>& samples) const {
    if (samples.size() < static_cast<std::size_t>(minimum_samples)) return true;
    if (samples.size() >= static_cast<std::size_t>(maximum_samples) || target_cv == 0) return false;
    const auto stats = analyze_timing(samples);
    return !stats.valid || stats.count < 3 || stats.cv > target_cv;
  }
};

} // namespace archbench
