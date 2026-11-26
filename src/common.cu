// SPDX-License-Identifier: AGPL-3.0-or-later
#include "archbench/common.cuh"
#include <chrono>

namespace archbench {

context::context(const options& input) : opts(input) {
  const auto requested_device = opts.integer("device", 0);
  if (requested_device < 0 || requested_device > std::numeric_limits<int>::max())
    throw std::invalid_argument("invalid device index");
  device = static_cast<int>(requested_device);
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::size_t total_bytes = 0;
  CUDA_CHECK(cudaMemGetInfo(&initial_free_bytes, &total_bytes));
}
context::~context() { if (stream) cudaStreamDestroy(stream); }

void context::memory_guard(std::size_t total_bytes) const {
  // This API receives TOTAL planned live allocations, even when called again
  // after base buffers were allocated. A stable budget avoids double-counting.
  const std::size_t free_bytes = initial_free_bytes;
  const double fraction = opts.real("memory-fraction", 0.70);
  if (fraction <= 0 || fraction > 0.95)
    throw std::invalid_argument("--memory-fraction must be in (0, 0.95]");
  const auto explicit_limit = opts.integer("memory-limit-mib", 0);
  if (explicit_limit < 0) throw std::invalid_argument("negative memory limit");
  std::size_t limit = static_cast<std::size_t>(static_cast<long double>(free_bytes) * fraction);
  if (explicit_limit) limit = std::min(limit, checked_product(static_cast<std::size_t>(explicit_limit), 1048576));
  if (total_bytes > limit)
    throw std::length_error("memory budget: requested " + std::to_string(total_bytes)
                          + " bytes, limit " + std::to_string(limit));
}

timing_policy context::measurement_policy() const {
  const auto samples_arg = opts.integer("samples", 7);
  const auto maximum_arg = opts.integer("max-samples", std::max<std::int64_t>(21, samples_arg));
  const auto warmup_arg = opts.integer("warmup", 3);
  const auto repeats_arg = opts.integer("max-repeats", 10000);
  const auto fixed_arg = opts.integer("repeats", 0);
  if (samples_arg < 1 || samples_arg > 10000 || maximum_arg < samples_arg || maximum_arg > 10000 ||
      warmup_arg < 0 || warmup_arg > 10000 ||
      repeats_arg < 1 || repeats_arg > 1000000 || fixed_arg < 0 || fixed_arg > repeats_arg)
    throw std::invalid_argument("invalid timing counts (fixed repeats must be within max-repeats)");
  timing_policy policy;
  policy.minimum_samples = static_cast<int>(samples_arg);
  policy.maximum_samples = static_cast<int>(maximum_arg);
  policy.warmup = static_cast<int>(warmup_arg);
  policy.max_repeats = static_cast<int>(repeats_arg);
  policy.repeats = static_cast<int>(fixed_arg);
  policy.target_ms = opts.real("target-ms", 20.0);
  policy.target_cv = opts.real("target-cv", .05);
  policy.warmup_ms = opts.real("warmup-ms", 0);
  if (policy.target_ms <= 0 || policy.target_ms > 60000 || policy.target_cv < 0 || policy.target_cv > 1 ||
      policy.warmup_ms < 0 || policy.warmup_ms > 60000)
    throw std::invalid_argument("invalid timing parameters");
  return policy;
}

void context::record_measurement(const std::vector<double>& samples, const std::string& method) {
  const auto policy = measurement_policy();
  const auto stats = analyze_timing(samples);
  timing_metadata["sampling_method"] = method;
  timing_metadata["sampling_quality_scope"] = "within-run dispersion of retained per-call sample averages; no outlier removal";
  timing_metadata["sampling_stop_reason"] = policy.target_cv == 0 ? "fixed_sample_count" :
      (!stats.valid || stats.count < 3) ? "insufficient_samples" :
      stats.cv <= policy.target_cv ? "target_cv_met" :
      stats.count >= static_cast<std::size_t>(policy.maximum_samples) ? "max_samples_reached" : "backend_fixed_count";
  timing_metrics["timing_minimum_samples"] = policy.minimum_samples;
  timing_metrics["timing_maximum_samples"] = policy.maximum_samples;
  timing_metrics["timing_target_cv"] = policy.target_cv;
  timing_metrics["timing_target_ms"] = policy.target_ms;
  timing_metrics["timing_warmup_target_host_ms"] = policy.warmup_ms;
  timing_metrics["timing_sample_count"] = samples.size();
  timing_metrics["timing_repeats"] = last_repeats;
  timing_metrics["timing_requested_fixed_repeats"] = policy.repeats;
  timing_metrics["timing_measured_calls"] = static_cast<double>(samples.size()) * last_repeats;
  if (stats.valid && stats.count > 1) timing_metrics["timing_observed_cv"] = stats.cv;
}

void context::annotate_timing(result& output) const {
  for (const auto& entry : timing_metrics) output.metrics[entry.first] = entry.second;
  for (const auto& entry : timing_metadata) output.metadata[entry.first] = entry.second;
  if (!output.samples_ms.empty() && !timing_metadata.count("sampling_method")) {
    output.metadata["sampling_method"] = "backend_fixed_count";
    output.metadata["sampling_stop_reason"] = "backend_fixed_count";
    output.metrics["timing_sample_count"] = output.samples_ms.size();
  }
}

std::vector<double> context::measure(const std::function<void()>& launch) {
  using clock = std::chrono::steady_clock;
  const auto policy = measurement_policy();
  timing_metrics.clear(); timing_metadata.clear();
  const auto warmup_start = clock::now();
  for (int i = 0; i < policy.warmup; ++i) launch();
  std::uint64_t warmup_calls = policy.warmup;
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  while (std::chrono::duration<double, std::milli>(clock::now() - warmup_start).count() < policy.warmup_ms) {
    launch(); ++warmup_calls;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  timing_metrics["warmup_host_ms"] = std::chrono::duration<double, std::milli>(clock::now() - warmup_start).count();
  timing_metrics["warmup_calls"] = static_cast<double>(warmup_calls);
  timing_metadata["warmup_scope"] = "minimum calls plus minimum host wall duration through stream completion";
  struct events {
    cudaEvent_t start = nullptr, stop = nullptr;
    events() {
      CUDA_CHECK(cudaEventCreate(&start));
      try { CUDA_CHECK(cudaEventCreate(&stop)); }
      catch (...) { cudaEventDestroy(start); throw; }
    }
    ~events() { cudaEventDestroy(start); cudaEventDestroy(stop); }
  } event;
  auto elapsed = [&](int count) {
    CUDA_CHECK(cudaEventRecord(event.start, stream));
    for (int i = 0; i < count; ++i) launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(event.stop, stream));
    CUDA_CHECK(cudaEventSynchronize(event.stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, event.start, event.stop));
    return static_cast<double>(milliseconds);
  };
  const auto calibration_start = clock::now();
  double calibration_device_ms = 0;
  int calibration_calls = 0;
  last_repeats = policy.repeats;
  if (!last_repeats) {
    const double calibration_ms = elapsed(1);
    calibration_device_ms = calibration_ms;
    calibration_calls = 1;
    const double desired = policy.target_ms / std::max(calibration_ms, 1e-6);
    last_repeats = static_cast<int>(std::max(1.0, std::min(static_cast<double>(policy.max_repeats), std::ceil(desired))));
    // A second calibration bounds severe first-call timing transients.
    if (last_repeats > 1) {
      const double check = elapsed(std::min(last_repeats, 16));
      calibration_device_ms += check;
      calibration_calls += std::min(last_repeats, 16);
      const double per_call = check / std::min(last_repeats, 16);
      last_repeats = static_cast<int>(std::max(1.0, std::min(static_cast<double>(policy.max_repeats),
                         std::ceil(policy.target_ms / std::max(per_call, 1e-6)))));
    }
  }
  timing_metadata["repetition_policy"] = policy.repeats ? "fixed" : "automatic_target_duration";
  timing_metrics["calibration_host_ms"] = std::chrono::duration<double, std::milli>(clock::now() - calibration_start).count();
  timing_metrics["calibration_device_ms"] = calibration_device_ms;
  timing_metrics["calibration_calls"] = calibration_calls;
  std::vector<double> times;
  times.reserve(policy.maximum_samples);
  double measured_device_ms = 0;
  const auto measurement_start = clock::now();
  while (policy.needs_more(times)) {
    const double batch_ms = elapsed(last_repeats);
    if (!std::isfinite(batch_ms) || batch_ms <= 0)
      throw std::runtime_error("nonpositive timing interval; increase target-ms/max-repeats and inspect the timer");
    measured_device_ms += batch_ms;
    times.push_back(batch_ms / last_repeats);
  }
  timing_metrics["measurement_host_ms"] = std::chrono::duration<double, std::milli>(clock::now() - measurement_start).count();
  timing_metrics["measured_device_ms"] = measured_device_ms;
  timing_metadata["timing_sample_scope"] = "CUDA event interval for repeated callbacks divided by repeats; enqueue gaps can contribute to the interval";
  record_measurement(times, "cuda_event_batch");
  return times;
}

std::string result::json() const {
  std::ostringstream out;
  out << "{\"schema_version\":1,\"status\":" << json_string(status)
      << ",\"validation\":" << json_string(validation)
      << ",\"reason\":" << json_string(reason)
      << ",\"execution\":" << json_string(execution)
      << ",\"unit\":" << json_string(unit) << ",\"work_per_iteration\":";
  json_number(out, work_per_iteration);
  out << ",\"bytes_per_iteration\":"; json_number(out, bytes_per_iteration);
  out << ",\"max_abs_error\":"; json_number(out, max_abs_error);
  out << ",\"max_rel_error\":"; json_number(out, max_rel_error);
  out << ",\"samples_ms\":[";
  for (std::size_t i = 0; i < samples_ms.size(); ++i) {
    if (i) out << ',';
    json_number(out, samples_ms[i]);
  }
  out << "],\"metadata\":{";
  bool first = true;
  for (const auto& [key, value] : metadata) {
    if (!first) out << ',';
    first = false;
    out << json_string(key) << ':' << json_string(value);
  }
  out << "},\"metrics\":{";
  auto all_metrics = metrics;
  const auto stats = analyze_timing(samples_ms);
  if (stats.valid) {
    const double median = stats.median;
    all_metrics["median_ms"] = median;
    all_metrics["median_us"] = median * 1e3;
    all_metrics["median_ns"] = median * 1e6;
    all_metrics["min_ms"] = stats.minimum;
    all_metrics["max_ms"] = stats.maximum;
    all_metrics["mean_ms"] = stats.mean;
    all_metrics["p10_batch_ms"] = stats.p10;
    all_metrics["p90_batch_ms"] = stats.p90;
    all_metrics["mad_ms"] = stats.mad;
    if (stats.count > 1) { all_metrics["stddev_ms"] = stats.stddev; all_metrics["cv"] = stats.cv; }
    if (stats.valid && median > 0) {
      if (work_per_iteration > 0) {
        all_metrics["useful_gops"] = work_per_iteration / (median * 1e6);
        all_metrics["useful_tera_work_s"] = work_per_iteration / (median * 1e9);
        all_metrics["useful_work_per_second"] = work_per_iteration / (median * 1e-3);
        all_metrics["useful_work_per_ns"] = work_per_iteration / (median * 1e6);
      }
      if (bytes_per_iteration > 0) {
        all_metrics["logical_gbytes_s"] = bytes_per_iteration / (median * 1e6);
        all_metrics["logical_gibytes_s"] = bytes_per_iteration / (median * 1e-3 * 1073741824.0);
      }
      if (work_per_iteration > 0 && bytes_per_iteration > 0)
        all_metrics["useful_work_per_logical_byte"] = work_per_iteration / bytes_per_iteration;
    }
  }
  first = true;
  for (const auto& [key, value] : all_metrics) {
    if (!first) out << ',';
    first = false;
    out << json_string(key) << ':'; json_number(out, value);
  }
  out << "}}";
  return out.str();
}
}  // namespace archbench
