// SPDX-License-Identifier: AGPL-3.0-or-later
#include "archbench/common.cuh"
#include <chrono>
#include <cublas_v2.h>
#include <cublasLt.h>
#ifdef ARCHBENCH_HAS_CUFFT
#include <cufft.h>
#endif
#ifdef ARCHBENCH_HAS_CUSPARSE
#include <cusparse.h>
#endif
#include <iostream>

#ifndef ARCHBENCH_TARGET_SM
#define ARCHBENCH_TARGET_SM 0
#endif
#ifndef ARCHBENCH_TARGET_NAME
#define ARCHBENCH_TARGET_NAME "unknown"
#endif
#ifndef ARCHBENCH_SOURCE_SHA256
#define ARCHBENCH_SOURCE_SHA256 "unavailable"
#endif
#ifndef ARCHBENCH_FLAGS_SHA256
#define ARCHBENCH_FLAGS_SHA256 "unavailable"
#endif
#ifndef ARCHBENCH_NVCC_VERSION
#define ARCHBENCH_NVCC_VERSION "unavailable"
#endif
#ifndef ARCHBENCH_VERSION
#define ARCHBENCH_VERSION "0.2.0"
#endif

using namespace archbench;

static options parse(int argc, char** argv) {
  options opts;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0) throw std::invalid_argument("expected --option: " + key);
    key = key.substr(2);
    const auto equals = key.find('=');
    if (equals != std::string::npos) {
      opts.values[key.substr(0, equals)] = key.substr(equals + 1);
    } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
      opts.values[key] = argv[++i];
    } else opts.values[key] = "true";
  }
  return opts;
}

static void identify(result& r, const context& ctx) {
  r.metadata["device_name"] = ctx.prop.name;
  r.metadata["compute_capability"] = std::to_string(ctx.prop.major) + "." + std::to_string(ctx.prop.minor);
  r.metadata["binary_target"] = ARCHBENCH_TARGET_NAME;
  r.metadata["archbench_version"] = ARCHBENCH_VERSION;
  r.metadata["source_sha256"] = ARCHBENCH_SOURCE_SHA256;
  r.metadata["build_flags_sha256"] = ARCHBENCH_FLAGS_SHA256;
  r.metadata["nvcc_version"] = ARCHBENCH_NVCC_VERSION;
  r.metadata["baseline_compilation"] = "project defaults: O3; lineinfo; no use_fast_math; no relocatable device code; externally supplied flags are included in build_flags_sha256";
  cublasHandle_t version_handle = nullptr;
  if (cublasCreate(&version_handle) == CUBLAS_STATUS_SUCCESS) {
    int version = 0;
    if (cublasGetVersion(version_handle, &version) == CUBLAS_STATUS_SUCCESS)
      r.metadata["cublas_version"] = std::to_string(version);
    cublasDestroy(version_handle);
  }
  r.metadata["cublaslt_version"] = std::to_string(cublasLtGetVersion());
#ifdef ARCHBENCH_HAS_CUFFT
  int fft_version = 0;
  r.metadata["cufft_version"] = cufftGetVersion(&fft_version) == CUFFT_SUCCESS ? std::to_string(fft_version) : "query_failed";
#else
  r.metadata["cufft_version"] = "disabled_at_build";
#endif
#ifdef ARCHBENCH_HAS_CUSPARSE
  cusparseHandle_t sparse_version_handle = nullptr;
  r.metadata["cusparse_version"] = "query_failed";
  if (cusparseCreate(&sparse_version_handle) == CUSPARSE_STATUS_SUCCESS) {
    int sparse_version = 0;
    if (cusparseGetVersion(sparse_version_handle, &sparse_version) == CUSPARSE_STATUS_SUCCESS)
      r.metadata["cusparse_version"] = std::to_string(sparse_version);
    cusparseDestroy(sparse_version_handle);
  }
#else
  r.metadata["cusparse_version"] = "disabled_at_build";
#endif
  r.metadata["sm_count"] = std::to_string(ctx.prop.multiProcessorCount);
  int visible_devices = 0;
  CUDA_CHECK(cudaGetDeviceCount(&visible_devices));
  r.metadata["visible_device_count"] = std::to_string(visible_devices);
  r.metadata["cuda_visible_ordinal"] = std::to_string(ctx.device);
#if defined(ARCHBENCH_HAS_FP64_MMA) && ARCHBENCH_HAS_FP64_MMA
  r.metadata["compile_probe.fp64_mma"] = "accepted";
#else
  r.metadata["compile_probe.fp64_mma"] = "not_accepted_or_not_probed";
#endif
#if defined(ARCHBENCH_SM120_INT4_MMA) && ARCHBENCH_SM120_INT4_MMA
  r.metadata["compile_probe.sm120_int4_mma"] = "accepted";
#else
  r.metadata["compile_probe.sm120_int4_mma"] = "not_accepted_or_not_applicable";
#endif
#if defined(ARCHBENCH_SM120_BINARY_MMA) && ARCHBENCH_SM120_BINARY_MMA
  r.metadata["compile_probe.sm120_binary_mma"] = "accepted";
#else
  r.metadata["compile_probe.sm120_binary_mma"] = "not_accepted_or_not_applicable";
#endif
  r.metadata["compile_probe_scope"] = "assembler acceptance and optional isolated-probe machine-code inspection; neither proves runtime support or measured throughput";
#ifdef ARCHBENCH_FP64_MMA_CODEGEN
  r.metadata["compile_probe.fp64_mma_codegen"] = ARCHBENCH_FP64_MMA_CODEGEN;
#endif
#ifdef ARCHBENCH_INT4_MMA_CODEGEN
  r.metadata["compile_probe.int4_mma_codegen"] = ARCHBENCH_INT4_MMA_CODEGEN;
#endif
#ifdef ARCHBENCH_BINARY_MMA_CODEGEN
  r.metadata["compile_probe.binary_mma_codegen"] = ARCHBENCH_BINARY_MMA_CODEGEN;
#endif
  r.metadata["total_memory_bytes"] = std::to_string(ctx.prop.totalGlobalMem);
  r.metadata["l2_bytes"] = std::to_string(ctx.prop.l2CacheSize);
  r.metadata["warp_size"] = std::to_string(ctx.prop.warpSize);
  r.metadata["memory_bus_width_bits"] = std::to_string(ctx.prop.memoryBusWidth);
  r.metadata["pci_domain"] = std::to_string(ctx.prop.pciDomainID);
  r.metadata["pci_bus"] = std::to_string(ctx.prop.pciBusID);
  r.metadata["pci_device"] = std::to_string(ctx.prop.pciDeviceID);
  std::ostringstream uuid;
  uuid << std::hex << std::setfill('0');
  for (unsigned char c : ctx.prop.uuid.bytes) uuid << std::setw(2) << unsigned(c);
  r.metadata["uuid_hex"] = uuid.str();
  int runtime = 0, driver = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  r.metadata["runtime_version"] = std::to_string(runtime);
  r.metadata["driver_api_version"] = std::to_string(driver);
  r.metadata["compile_cudart_version"] = std::to_string(CUDART_VERSION);
  r.metadata["timing"] = "CUDA events on the execution stream; median of per-call batch averages";
  r.metadata["sass_evidence"] = "not_collected_by_worker; use bench disassemble/profile";
}

int main(int argc, char** argv) {
  result r;
  const auto start = std::chrono::steady_clock::now();
  try {
    auto opts = parse(argc, argv);
    if (opts.flag("help")) {
      std::cout << "archbench CUDA worker " << ARCHBENCH_VERSION << " target " << ARCHBENCH_TARGET_NAME
        << "\nUse ./bench list or ./bench run. Direct: --family scalar --op fma --dtype f32\n"
        << "Controls: --device 0 --samples 7 --max-samples 21 --target-cv 0.05 --warmup 3 --warmup-ms 0 --target-ms 20 --repeats 0 --max-repeats 10000\n"
        << "Memory: --memory-fraction 0.70 --memory-limit-mib 0\n";
      return 0;
    }
    context ctx(opts);
    identify(r, ctx);
    for (const auto& [key, value] : opts.values) r.metadata["option." + key] = value;
    if (opts.flag("info")) {
      r.validation = "not_applicable";
      r.execution = "metadata_only";
    } else if (ARCHBENCH_TARGET_SM && ctx.prop.major * 10 + ctx.prop.minor != ARCHBENCH_TARGET_SM) {
      skip(r, "binary target differs from this GPU; select a binary compiled for the exact architecture", "architecture_mismatch");
    } else {
      if (!(run_alu(ctx, r) || run_gemm(ctx, r) || run_systems_software(ctx, r) || run_systems(ctx, r) || run_quant(ctx, r)))
        skip(r, "family not implemented by this worker", "not_implemented");
      r.metadata["timing_repeats"] = std::to_string(ctx.last_repeats);
      CUDA_CHECK(cudaDeviceSynchronize());
      ctx.annotate_timing(r);
      if (r.status == "ok" && r.validation == "not_applicable") r.status = "measurement_only";
      if (r.status == "ok" && r.validation == "failed") r.status = "validation_failed";
      if (r.status == "ok" && r.validation == "not_checked") {
        r.status = "not_validated";
        r.reason = "backend did not provide numerical validation";
      }
      if (r.status == "ok" && r.samples_ms.empty()) {
        r.status = "error";
        r.reason = "backend returned no timing samples";
      }
    }
  } catch (const cuda_error& error) {
    r.status = error.code == cudaErrorMemoryAllocation ? "out_of_memory" : "error";
    r.reason = error.what();
    r.validation = "not_run";
  } catch (const std::length_error& error) {
    skip(r, error.what(), "memory_budget");
  } catch (const std::exception& error) {
    r.status = "error";
    r.reason = error.what();
    r.validation = "not_run";
  }
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  r.metrics["worker_wall_seconds"] = elapsed;
  std::cout << r.json() << '\n';
  return r.status == "error" || r.status == "validation_failed" ? 1 : 0;
}
