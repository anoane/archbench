// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include <cuda_runtime.h>
#include "archbench/timing_stats.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace archbench {

struct options {
  std::map<std::string, std::string> values;
  std::string get(const std::string& key, const std::string& fallback = "") const {
    const auto it = values.find(key);
    return it == values.end() ? fallback : it->second;
  }
  bool has(const std::string& key) const { return values.count(key) != 0; }
  std::int64_t integer(const std::string& key, std::int64_t fallback) const {
    if (!has(key)) return fallback;
    std::size_t end = 0;
    const auto value = std::stoll(get(key), &end, 0);
    if (end != get(key).size()) throw std::invalid_argument("invalid integer: --" + key);
    return value;
  }
  double real(const std::string& key, double fallback) const {
    if (!has(key)) return fallback;
    std::size_t end = 0;
    const auto value = std::stod(get(key), &end);
    if (end != get(key).size() || !std::isfinite(value))
      throw std::invalid_argument("invalid finite number: --" + key);
    return value;
  }
  bool flag(const std::string& key, bool fallback = false) const {
    if (!has(key)) return fallback;
    const auto value = get(key);
    if (value == "true" || value == "1" || value == "yes") return true;
    if (value == "false" || value == "0" || value == "no") return false;
    throw std::invalid_argument("invalid boolean: --" + key);
  }
};

inline std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) out << "\\u" << std::hex << std::setw(4)
                          << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
  }
  out << '"';
  return out.str();
}

inline void json_number(std::ostream& out, double number) {
  if (std::isfinite(number)) out << std::setprecision(17) << number;
  else out << "null";
}

struct result {
  std::string status = "ok";
  std::string validation = "not_checked";
  std::string reason;
  // PTX is a virtual ISA: native execution is only asserted after SASS inspection.
  std::string execution = "compiler_lowered";
  std::string unit = "Gop/s";
  double work_per_iteration = 0;
  double bytes_per_iteration = 0;
  double max_abs_error = 0;
  double max_rel_error = 0;
  std::vector<double> samples_ms;
  std::map<std::string, std::string> metadata;
  std::map<std::string, double> metrics;
  std::string json() const;
};

inline void skip(result& r, const std::string& reason,
                 const std::string& status = "unsupported") {
  r.status = status;
  r.reason = reason;
  r.validation = "not_run";
  r.execution = "not_executed";
  r.samples_ms.clear();
}

class cuda_error : public std::runtime_error {
 public:
  cudaError_t code;
  cuda_error(cudaError_t code, const std::string& message)
      : std::runtime_error(message), code(code) {}
};

inline void cuda_check(cudaError_t code, const char* expression,
                       const char* file, int line) {
  if (code != cudaSuccess)
    throw cuda_error(code, std::string(expression) + ": " + cudaGetErrorString(code)
                   + " (" + file + ":" + std::to_string(line) + ")");
}
#define CUDA_CHECK(expression) \
  ::archbench::cuda_check((expression), #expression, __FILE__, __LINE__)

inline std::size_t checked_product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("allocation size overflow");
  return a * b;
}

template<class T> class device_buffer {
  T* pointer_ = nullptr;
  std::size_t count_ = 0;
 public:
  explicit device_buffer(std::size_t count) : count_(count) {
    if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                                    checked_product(count, sizeof(T))));
  }
  device_buffer() = default;
  ~device_buffer() { if (pointer_) cudaFree(pointer_); }
  device_buffer(const device_buffer&) = delete;
  device_buffer& operator=(const device_buffer&) = delete;
  device_buffer(device_buffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)),
        count_(std::exchange(other.count_, 0)) {}
  device_buffer& operator=(device_buffer&& other) noexcept {
    if (this != &other) {
      if (pointer_) cudaFree(pointer_);
      pointer_ = std::exchange(other.pointer_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  T* get() { return pointer_; }
  const T* get() const { return pointer_; }
  std::size_t size() const { return count_; }
  std::size_t bytes() const { return checked_product(count_, sizeof(T)); }
};

template<class T> inline std::vector<T> read_buffer(const device_buffer<T>& b) {
  std::vector<T> host(b.size());
  if (b.size()) CUDA_CHECK(cudaMemcpy(host.data(), b.get(), b.bytes(), cudaMemcpyDeviceToHost));
  return host;
}

struct context {
  const options& opts;
  cudaDeviceProp prop{};
  cudaStream_t stream = nullptr;
  int device = 0;
  int last_repeats = 1;
  std::size_t initial_free_bytes = 0;
  std::map<std::string, double> timing_metrics;
  std::map<std::string, std::string> timing_metadata;
  explicit context(const options& opts);
  ~context();
  context(const context&) = delete;
  context& operator=(const context&) = delete;
  std::vector<double> measure(const std::function<void()>& launch);
  timing_policy measurement_policy() const;
  void record_measurement(const std::vector<double>& samples, const std::string& method);
  void annotate_timing(result& output) const;
  void memory_guard(std::size_t total_bytes) const;
};

bool run_alu(context&, result&);
bool run_gemm(context&, result&);
bool run_systems(context&, result&);
bool run_systems_software(context&, result&);
bool run_quant(context&, result&);

}  // namespace archbench
