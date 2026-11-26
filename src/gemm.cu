// SPDX-License-Identifier: AGPL-3.0-or-later
// GEMM/BLAS experiments. Library selection is deliberately not called native.
#include "archbench/common.cuh"
#include "archbench/blas_software.hpp"
#include "gemm_capabilities.hpp"
#include "gemm_software_model.hpp"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuComplex.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace archbench {
namespace gemm_detail {

using wide = std::complex<long double>;
using clock_type = std::chrono::steady_clock;

struct library_failure : std::runtime_error {
    cublasStatus_t status;
    library_failure(cublasStatus_t s, const char* expression)
        : std::runtime_error(std::string(expression) + " returned cuBLAS status " +
                             std::to_string(static_cast<int>(s))), status(s) {}
};
inline void check(cublasStatus_t s, const char* expression) {
    if (s != CUBLAS_STATUS_SUCCESS) throw library_failure(s, expression);
}
#define BLAS_CHECK(expression) ::archbench::gemm_detail::check((expression), #expression)

inline size_t product(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::invalid_argument("matrix size overflows size_t");
    return a * b;
}
inline size_t sum_sizes(std::initializer_list<size_t> values) {
    size_t result = 0;
    for (const auto value : values) {
        if (value > std::numeric_limits<size_t>::max() - result)
            throw std::invalid_argument("allocation byte total overflows size_t");
        result += value;
    }
    return result;
}
inline int bounded_int(const options& opts, const std::string& key, int fallback,
                       int minimum, int maximum) {
    const auto value = opts.integer(key, fallback);
    if (value < minimum || value > maximum)
        throw std::invalid_argument(key + " must be in [" + std::to_string(minimum) +
                                    ", " + std::to_string(maximum) + "]");
    return static_cast<int>(value);
}
inline int positive_int(const options& opts, const std::string& key, int fallback) {
    return bounded_int(opts,key,fallback,1,std::numeric_limits<int>::max());
}
inline cublasOperation_t transpose(const std::string& s) {
    if (s == "N" || s == "n") return CUBLAS_OP_N;
    if (s == "T" || s == "t") return CUBLAS_OP_T;
    if (s == "C" || s == "c") return CUBLAS_OP_C;
    throw std::invalid_argument("transpose must be N, T, or C");
}

struct handle {
    cublasHandle_t value = nullptr;
    explicit handle(cudaStream_t stream) {
        BLAS_CHECK(cublasCreate(&value));
        try { BLAS_CHECK(cublasSetStream(value, stream)); }
        catch (...) { cublasDestroy(value); value = nullptr; throw; }
    }
    ~handle() { if (value) cublasDestroy(value); }
    handle(const handle&) = delete;
    handle& operator=(const handle&) = delete;
};
struct lt_handle {
    cublasLtHandle_t value = nullptr;
    lt_handle() { BLAS_CHECK(cublasLtCreate(&value)); }
    ~lt_handle() { if (value) cublasLtDestroy(value); }
};
struct lt_layout {
    cublasLtMatrixLayout_t value = nullptr;
    lt_layout(cudaDataType_t t, uint64_t rows, uint64_t cols, int64_t ld) {
        BLAS_CHECK(cublasLtMatrixLayoutCreate(&value, t, rows, cols, ld));
    }
    ~lt_layout() { if (value) cublasLtMatrixLayoutDestroy(value); }
    template<class T> void set(cublasLtMatrixLayoutAttribute_t key, const T& x) {
        BLAS_CHECK(cublasLtMatrixLayoutSetAttribute(value, key, &x, sizeof(x)));
    }
};
struct lt_descriptor {
    cublasLtMatmulDesc_t value = nullptr;
    lt_descriptor(cublasComputeType_t compute, cudaDataType_t scale) {
        BLAS_CHECK(cublasLtMatmulDescCreate(&value, compute, scale));
    }
    ~lt_descriptor() { if (value) cublasLtMatmulDescDestroy(value); }
    template<class T> void set(cublasLtMatmulDescAttributes_t key, const T& x) {
        BLAS_CHECK(cublasLtMatmulDescSetAttribute(value, key, &x, sizeof(x)));
    }
};
struct lt_preference {
    cublasLtMatmulPreference_t value = nullptr;
    lt_preference() { BLAS_CHECK(cublasLtMatmulPreferenceCreate(&value)); }
    ~lt_preference() { if (value) cublasLtMatmulPreferenceDestroy(value); }
};
#if CUDART_VERSION >= 13050
struct lt_emulation {
    cublasLtEmulationDesc_t value = nullptr;
    lt_emulation() { BLAS_CHECK(cublasLtEmulationDescCreate(&value)); }
    ~lt_emulation() { if(value) cublasLtEmulationDescDestroy(value); }
    template<class T> void set(cublasLtEmulationDescAttributes_t attribute,const T& value_to_set) {
        BLAS_CHECK(cublasLtEmulationDescSetAttribute(value,attribute,&value_to_set,sizeof(value_to_set)));
    }
};
inline cublasEmulationStrategy_t emulation_strategy(const options& opts) {
    const auto s=opts.get("emulation-strategy","eager");
    if(s=="eager")return CUBLAS_EMULATION_STRATEGY_EAGER;
    if(s=="performant")return CUBLAS_EMULATION_STRATEGY_PERFORMANT;
    if(s=="default")return CUBLAS_EMULATION_STRATEGY_DEFAULT;
    throw std::invalid_argument("emulation-strategy must be eager, performant, or default");
}
inline cudaEmulationMantissaControl_t mantissa_control(const options& opts) {
    const auto s=opts.get("mantissa-control","dynamic");
    if(s=="dynamic")return CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC;
    if(s=="fixed")return CUDA_EMULATION_MANTISSA_CONTROL_FIXED;
    throw std::invalid_argument("mantissa-control must be dynamic or fixed");
}
#endif
struct event_pair {
    cudaEvent_t begin = nullptr, end = nullptr;
    event_pair() {
        CUDA_CHECK(cudaEventCreate(&begin));
        try { CUDA_CHECK(cudaEventCreate(&end)); }
        catch (...) { cudaEventDestroy(begin); throw; }
    }
    ~event_pair() { if (begin) cudaEventDestroy(begin); if (end) cudaEventDestroy(end); }
};

// In-place BLAS calls require restored input between invocations. Their reset is
// enqueued BEFORE the start event and is excluded from the reported operation.
// Individual CUDA event intervals are summed, rather than timing all resets.
inline std::vector<double> measure_restored(context& ctx,
        const std::function<void()>& restore, const std::function<void()>& launch) {
    const auto policy=ctx.measurement_policy();
    if(policy.repeats>2048)
        throw std::invalid_argument("restored-input timing supports at most 2048 explicit repeats because every call needs its own event pair");
    ctx.timing_metrics.clear();ctx.timing_metadata.clear();
    const auto warmup_start=clock_type::now();
    size_t warmup_calls=0;
    for (int i=0;i<policy.warmup;++i) {restore();launch();++warmup_calls;}
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    while(std::chrono::duration<double,std::milli>(clock_type::now()-warmup_start).count()<policy.warmup_ms) {
        restore();launch();++warmup_calls;CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    }
    ctx.timing_metrics["warmup_calls"]=static_cast<double>(warmup_calls);
    ctx.timing_metrics["warmup_host_ms"]=std::chrono::duration<double,std::milli>(clock_type::now()-warmup_start).count();
    ctx.timing_metadata["warmup_scope"]="host wall time including restore calls and stream completion";
    const auto calibration_start=clock_type::now();
    event_pair calibration;
    float ms = 0;
    int repeats=policy.repeats;
    if(!repeats) {
        restore();CUDA_CHECK(cudaEventRecord(calibration.begin,ctx.stream));launch();
        CUDA_CHECK(cudaEventRecord(calibration.end,ctx.stream));CUDA_CHECK(cudaEventSynchronize(calibration.end));
        CUDA_CHECK(cudaEventElapsedTime(&ms,calibration.begin,calibration.end));
        const int cap=std::min(policy.max_repeats,2048);
        repeats=std::clamp(static_cast<int>(std::min<double>(cap,
            std::ceil(policy.target_ms/std::max(1e-6,static_cast<double>(ms))))),1,cap);
    }
    ctx.timing_metrics["calibration_device_ms"]=ms;
    ctx.timing_metrics["calibration_calls"]=policy.repeats?0:1;
    ctx.timing_metrics["calibration_host_ms"]=std::chrono::duration<double,std::milli>(clock_type::now()-calibration_start).count();
    ctx.timing_metadata["repetition_policy"]=policy.repeats?"fixed":"automatic_target_duration";
    ctx.last_repeats = repeats;
    std::vector<cudaEvent_t> starts(repeats, nullptr), stops(repeats, nullptr);
    auto cleanup = [&] {
        for (auto e : starts) if (e) cudaEventDestroy(e);
        for (auto e : stops) if (e) cudaEventDestroy(e);
    };
    std::vector<double> out;
    double measured_device_ms=0;
    auto measurement_start=clock_type::now();
    try {
        for (int i = 0; i < repeats; ++i) {
            CUDA_CHECK(cudaEventCreate(&starts[i]));
            CUDA_CHECK(cudaEventCreate(&stops[i]));
        }
        measurement_start=clock_type::now();
        while(policy.needs_more(out)) {
            for (int i = 0; i < repeats; ++i) {
                restore();
                CUDA_CHECK(cudaEventRecord(starts[i], ctx.stream));
                launch();
                CUDA_CHECK(cudaEventRecord(stops[i], ctx.stream));
            }
            CUDA_CHECK(cudaEventSynchronize(stops.back()));
            double sum = 0;
            for (int i = 0; i < repeats; ++i) {
                CUDA_CHECK(cudaEventElapsedTime(&ms, starts[i], stops[i]));
                if(!std::isfinite(ms)||ms<=0)
                    throw std::runtime_error("nonpositive per-call restored-operation event interval; inspect timer resolution and use a larger problem");
                sum += ms;
            }
            if(!std::isfinite(sum)||sum<=0)
                throw std::runtime_error("nonpositive restored-operation timing interval; increase repetitions and inspect the timer");
            measured_device_ms+=sum;
            out.push_back(sum / repeats);
        }
    } catch (...) { cleanup(); throw; }
    ctx.timing_metrics["measurement_host_ms"]=std::chrono::duration<double,std::milli>(clock_type::now()-measurement_start).count();
    ctx.timing_metrics["measured_device_ms"]=measured_device_ms;
    ctx.timing_metadata["timing_sample_scope"]="sum of per-call CUDA event intervals divided by repeats; every input reset is queued before that call's start event";
    ctx.record_measurement(out,"individual_cuda_events_excluding_reset");
    cleanup();
    return out;
}

using kind = gemm_software::storage_kind;
struct type_info {
    kind id;
    cudaDataType_t cuda;
    int bits;
    bool complex;
    bool narrow;
};
inline type_info parse_type(const std::string& name) {
    if (name == "f16") return {kind::f16, CUDA_R_16F, 16, false, false};
    if (name == "bf16") return {kind::bf16, CUDA_R_16BF, 16, false, false};
    if (name == "f32") return {kind::f32, CUDA_R_32F, 32, false, false};
    if (name == "f64") return {kind::f64, CUDA_R_64F, 64, false, false};
    if (name == "c64") return {kind::c64, CUDA_C_32F, 64, true, false};
    if (name == "c128") return {kind::c128, CUDA_C_64F, 128, true, false};
    if (name == "i8") return {kind::i8, CUDA_R_8I, 8, false, false};
    if (name == "i32") return {kind::i32, CUDA_R_32I, 32, false, false};
    if (name == "fp8_e4m3") return {kind::e4m3, CUDA_R_8F_E4M3, 8, false, true};
    if (name == "fp8_e5m2") return {kind::e5m2, CUDA_R_8F_E5M2, 8, false, true};
#if CUDART_VERSION >= 12080
    if (name == "fp4_e2m1" || name == "nvfp4")
        return {kind::e2m1, CUDA_R_4F_E2M1, 4, false, true};
#endif
    throw std::invalid_argument("unknown GEMM data type: " + name);
}
inline size_t storage_bytes(size_t elements, type_info t) {
    const size_t bits = product(elements, static_cast<size_t>(t.bits));
    return bits / 8 + (bits % 8 != 0);
}

__host__ __device__ inline uint32_t hash_index(uint64_t i, uint32_t seed) {
    uint32_t x = static_cast<uint32_t>(i) ^ static_cast<uint32_t>(i >> 32) ^ seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
__host__ __device__ inline double generated_value(uint64_t i, uint32_t seed) {
    const int v = static_cast<int>(hash_index(i, seed) % 2003u) - 1001;
    return static_cast<double>(v) / 2003.0;
}
template<class T> __host__ __device__ T make_value(uint64_t i, uint32_t seed) {
    return static_cast<T>(generated_value(i, seed));
}
template<> __host__ __device__ inline __half make_value<__half>(uint64_t i, uint32_t s) {
    return __float2half_rn(static_cast<float>(generated_value(i, s)));
}
template<> __host__ __device__ inline __nv_bfloat16 make_value<__nv_bfloat16>(uint64_t i, uint32_t s) {
    return __float2bfloat16_rn(static_cast<float>(generated_value(i, s)));
}
template<> __host__ __device__ inline cuComplex make_value<cuComplex>(uint64_t i, uint32_t s) {
    return make_cuComplex(static_cast<float>(generated_value(i, s)),
                          static_cast<float>(generated_value(i, s ^ 0x5a17u)));
}
template<> __host__ __device__ inline cuDoubleComplex make_value<cuDoubleComplex>(uint64_t i, uint32_t s) {
    return make_cuDoubleComplex(generated_value(i, s), generated_value(i, s ^ 0x5a17u));
}
template<> __host__ __device__ inline int8_t make_value<int8_t>(uint64_t i, uint32_t s) {
    return static_cast<int8_t>(static_cast<int>(hash_index(i, s) % 7u) - 3);
}
template<> __host__ __device__ inline int32_t make_value<int32_t>(uint64_t i, uint32_t s) {
    return static_cast<int32_t>(hash_index(i, s) % 7u) - 3;
}
template<> __host__ __device__ inline __nv_fp8_e4m3 make_value<__nv_fp8_e4m3>(uint64_t i, uint32_t s) {
    return __nv_fp8_e4m3(static_cast<float>(generated_value(i, s)));
}
template<> __host__ __device__ inline __nv_fp8_e5m2 make_value<__nv_fp8_e5m2>(uint64_t i, uint32_t s) {
    return __nv_fp8_e5m2(static_cast<float>(generated_value(i, s)));
}

template<class T> __host__ __device__ double2 as_double2(T x) {
    return make_double2(static_cast<double>(x), 0.0);
}
template<> __host__ __device__ inline double2 as_double2(__half x) {
    return make_double2(__half2float(x), 0.0);
}
template<> __host__ __device__ inline double2 as_double2(__nv_bfloat16 x) {
    return make_double2(__bfloat162float(x), 0.0);
}
template<> __host__ __device__ inline double2 as_double2(__nv_fp8_e4m3 x) {
    return make_double2(static_cast<float>(x), 0.0);
}
template<> __host__ __device__ inline double2 as_double2(__nv_fp8_e5m2 x) {
    return make_double2(static_cast<float>(x), 0.0);
}
template<> __host__ __device__ inline double2 as_double2(cuComplex x) {
    return make_double2(cuCrealf(x), cuCimagf(x));
}
template<> __host__ __device__ inline double2 as_double2(cuDoubleComplex x) {
    return make_double2(cuCreal(x), cuCimag(x));
}
template<class T> inline wide as_wide(T x) {
    const auto d = as_double2(x); return {d.x, d.y};
}
inline wide generated_wide(kind t, uint64_t index, uint32_t seed) {
    switch (t) {
    case kind::f16: return as_wide(make_value<__half>(index, seed));
    case kind::bf16: return as_wide(make_value<__nv_bfloat16>(index, seed));
    case kind::f32: return as_wide(make_value<float>(index, seed));
    case kind::f64: return as_wide(make_value<double>(index, seed));
    case kind::c64: return as_wide(make_value<cuComplex>(index, seed));
    case kind::c128: return as_wide(make_value<cuDoubleComplex>(index, seed));
    case kind::i8: return as_wide(make_value<int8_t>(index, seed));
    case kind::i32: return as_wide(make_value<int32_t>(index, seed));
    case kind::e4m3: return as_wide(make_value<__nv_fp8_e4m3>(index, seed));
    case kind::e5m2: return as_wide(make_value<__nv_fp8_e5m2>(index, seed));
    case kind::e2m1: {
        static constexpr double magnitudes[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
        const unsigned v = hash_index(index, seed) & 15;
        return {magnitudes[v & 7] * ((v & 8) ? -1 : 1), 0};
    }
    }
    return {};
}
template<class T> __global__ void fill_kernel(T* p, size_t n, uint32_t seed) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        p[i] = make_value<T>(i, seed);
}
__global__ void fill_fp4_kernel(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < (n + 1) / 2; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint8_t lo = hash_index(2 * i, seed) & 15;
        const uint8_t hi = 2 * i + 1 < n ? (hash_index(2 * i + 1, seed) & 15) : 0;
        p[i] = lo | (hi << 4);
    }
}
inline void fill_buffer(context& ctx, void* p, type_info type, size_t count, uint32_t seed) {
    const int blocks = static_cast<int>(std::min<size_t>((count + 255) / 256, 4096));
#define FILL_CASE(k, T) case kind::k: fill_kernel<T><<<blocks, 256, 0, ctx.stream>>>(static_cast<T*>(p), count, seed); break
    switch (type.id) {
    FILL_CASE(f16, __half); FILL_CASE(bf16, __nv_bfloat16); FILL_CASE(f32, float);
    FILL_CASE(f64, double); FILL_CASE(c64, cuComplex); FILL_CASE(c128, cuDoubleComplex);
    FILL_CASE(i8, int8_t); FILL_CASE(i32, int32_t); FILL_CASE(e4m3, __nv_fp8_e4m3);
    FILL_CASE(e5m2, __nv_fp8_e5m2);
    case kind::e2m1: fill_fp4_kernel<<<blocks, 256, 0, ctx.stream>>>(static_cast<uint8_t*>(p), count, seed); break;
    }
#undef FILL_CASE
    CUDA_CHECK(cudaGetLastError());
}

struct problem {
    int m, n, k, batch, lda, ldb, ldc;
    cublasOperation_t ta, tb;
    size_t stride_a, stride_b, stride_c, count_a, count_b, count_c;
    explicit problem(const options& opts) {
        m = positive_int(opts, "m", 1024); n = positive_int(opts, "n", 1024);
        k = positive_int(opts, "k", 1024); batch = positive_int(opts, "batch", 1);
        ta = transpose(opts.get("trans-a", "N")); tb = transpose(opts.get("trans-b", "N"));
        const auto pad64 = opts.integer("ld-pad", 0);
        if (pad64 < 0 || pad64 > std::numeric_limits<int>::max() - std::max({m,n,k}))
            throw std::invalid_argument("ld-pad is negative or too large");
        const int pad = static_cast<int>(pad64);
        lda = (ta == CUBLAS_OP_N ? m : k) + pad;
        ldb = (tb == CUBLAS_OP_N ? k : n) + pad;
        ldc = m + pad;
        const auto gap64 = opts.integer("batch-gap", 0);
        if (gap64 < 0) throw std::invalid_argument("batch-gap cannot be negative");
        const size_t gap = static_cast<size_t>(gap64);
        const auto matrix_stride = [gap](size_t ld,size_t columns) {
            const auto area = product(ld,columns);
            const auto maximum = static_cast<size_t>(std::numeric_limits<int64_t>::max());
            if (gap > maximum || area > maximum - gap)
                throw std::invalid_argument("matrix size plus batch-gap exceeds the signed 64-bit library stride");
            return area + gap;
        };
        stride_a = matrix_stride(lda, ta == CUBLAS_OP_N ? k : m);
        stride_b = matrix_stride(ldb, tb == CUBLAS_OP_N ? n : k);
        stride_c = matrix_stride(ldc,n);
        count_a = product(stride_a, batch); count_b = product(stride_b, batch);
        count_c = product(stride_c, batch);
    }
    size_t ai(int row, int reduction, int b) const {
        return b * stride_a + (ta == CUBLAS_OP_N ? reduction * size_t(lda) + row : row * size_t(lda) + reduction);
    }
    size_t bi(int reduction, int column, int b) const {
        return b * stride_b + (tb == CUBLAS_OP_N ? column * size_t(ldb) + reduction : reduction * size_t(ldb) + column);
    }
};

inline bool rank1_problem(const context& ctx,const problem& p,type_info a,type_info b,type_info c) {
    const auto pattern=ctx.opts.get("input-pattern","seeded");
    if(pattern!="seeded"&&pattern!="rank1")throw std::invalid_argument("input-pattern must be seeded or rank1");
    if(pattern=="seeded")return false;
    if((a.id!=kind::f16&&a.id!=kind::bf16)||b.id!=a.id||c.id!=kind::f32||
       p.ta!=CUBLAS_OP_N||p.tb!=CUBLAS_OP_N||p.batch!=1||ctx.opts.real("alpha",1)!=1||ctx.opts.real("beta",0)!=0||
       ctx.opts.real("alpha-imag",0)!=0||ctx.opts.real("beta-imag",0)!=0||ctx.opts.get("epilogue","default")!="default"||
       ctx.opts.get("compute","default")!="default"||ctx.opts.get("scale-mode","tensor")!="tensor"||
       ctx.opts.real("scale-a",1)!=1||ctx.opts.real("scale-b",1)!=1||ctx.opts.flag("fast-accum",false))
        throw std::invalid_argument("rank1 requires matching f16 or bf16 inputs, f32 output, NN, batch=1, alpha=1, beta=0, default compute/epilogue, and no input scaling");
    // Every partial sum of these binary fractions is exactly representable in
    // FP32 over this bounded K range; do not extend equality to arbitrary K.
    if(p.k>65536)throw std::invalid_argument("rank1 exact FP32 reference requires K<=65536");
    return true;
}
template<class T> __global__ void fill_rank1_kernel(T* data,size_t count,int rows,int ld,bool operand_a) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=size_t(gridDim.x)*blockDim.x) {
        const int row=int(i%ld),column=int(i/ld);
        const float value=row<rows?gemm_software::rank1_input(operand_a,row,column):0;
        if constexpr(std::is_same_v<T,__half>)data[i]=__float2half_rn(value);
        else data[i]=__float2bfloat16_rn(value);
    }
}
inline void fill_gemm_inputs(context& ctx,const problem& p,type_info a,type_info b,type_info c,
                             void* da,void* db,void* dc) {
    if(rank1_problem(ctx,p,a,b,c)) {
        const int blocks_a=int(std::min<size_t>((p.count_a+255)/256,4096));
        const int blocks_b=int(std::min<size_t>((p.count_b+255)/256,4096));
        if(a.id==kind::f16) {
            fill_rank1_kernel<<<blocks_a,256,0,ctx.stream>>>(static_cast<__half*>(da),p.count_a,p.m,p.lda,true);
            fill_rank1_kernel<<<blocks_b,256,0,ctx.stream>>>(static_cast<__half*>(db),p.count_b,p.k,p.ldb,false);
        } else {
            fill_rank1_kernel<<<blocks_a,256,0,ctx.stream>>>(static_cast<__nv_bfloat16*>(da),p.count_a,p.m,p.lda,true);
            fill_rank1_kernel<<<blocks_b,256,0,ctx.stream>>>(static_cast<__nv_bfloat16*>(db),p.count_b,p.k,p.ldb,false);
        }
        CUDA_CHECK(cudaGetLastError());
    } else {fill_buffer(ctx,da,a,p.count_a,17);fill_buffer(ctx,db,b,p.count_b,29);}
    fill_buffer(ctx,dc,c,p.count_c,71);
}

struct scalar_pair {
    float af, bf; double ad, bd; int32_t ai, bi; __half ah, bh;
    cuComplex ac, bc; cuDoubleComplex az, bz;
    wide alpha, beta;
    cudaDataType_t type;
    const void *a = nullptr, *b = nullptr;
    scalar_pair(const options& opts, type_info output, cublasComputeType_t compute) {
        const double x = opts.real("alpha", 1), y = opts.real("beta", 0);
        const double xi = opts.real("alpha-imag", 0), yi = opts.real("beta-imag", 0);
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(xi) || !std::isfinite(yi))
            throw std::invalid_argument("alpha and beta must be finite");
        if (!output.complex && (xi != 0 || yi != 0))
            throw std::invalid_argument("imaginary alpha/beta require complex output");
        ad = x; bd = y; af = static_cast<float>(x); bf = static_cast<float>(y);
        ah = __float2half_rn(af); bh = __float2half_rn(bf);
        ac = make_cuComplex(af, static_cast<float>(xi)); bc = make_cuComplex(bf, static_cast<float>(yi));
        az = make_cuDoubleComplex(ad, xi); bz = make_cuDoubleComplex(bd, yi);
        if (output.id == kind::c64) { type = CUDA_C_32F; a = &ac; b = &bc; alpha = as_wide(ac); beta = as_wide(bc); }
        else if (output.id == kind::c128) { type = CUDA_C_64F; a = &az; b = &bz; alpha = as_wide(az); beta = as_wide(bz); }
        else if (output.id == kind::f64) { type = CUDA_R_64F; a = &ad; b = &bd; alpha = ad; beta = bd; }
        else if (compute == CUBLAS_COMPUTE_16F || compute == CUBLAS_COMPUTE_16F_PEDANTIC) {
            type = CUDA_R_16F; a = &ah; b = &bh; alpha = as_wide(ah); beta = as_wide(bh);
        } else if (output.id == kind::i32) {
            if (x != std::trunc(x) || y != std::trunc(y) || std::abs(x) > 1024 || std::abs(y) > 1024)
                throw std::invalid_argument("integer alpha/beta must be small exact integers");
            ai = static_cast<int32_t>(x); bi = static_cast<int32_t>(y);
            type = CUDA_R_32I; a = &ai; b = &bi; alpha = ai; beta = bi;
        } else { type = CUDA_R_32F; a = &af; b = &bf; alpha = af; beta = bf; }
    }
};

inline cublasComputeType_t compute_type(const options& opts, type_info in) {
    const std::string s = opts.get("compute", "default");
    if (s == "f16") return CUBLAS_COMPUTE_16F;
    if (s == "f16_pedantic") return CUBLAS_COMPUTE_16F_PEDANTIC;
    if (s == "tf32") return CUBLAS_COMPUTE_32F_FAST_TF32;
    if (s == "fast_f16") return CUBLAS_COMPUTE_32F_FAST_16F;
    if (s == "fast_bf16") return CUBLAS_COMPUTE_32F_FAST_16BF;
#if CUDART_VERSION >= 13050
    if (s == "bf16x9") return CUBLAS_COMPUTE_32F_EMULATED_16BFX9;
    if (s == "fixedpoint") return CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
#endif
    if (s != "default" && s != "pedantic") throw std::invalid_argument("unknown compute mode: " + s);
    const bool pedantic = s == "pedantic";
    if (in.id == kind::f64 || in.id == kind::c128)
        return pedantic ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F;
    if (in.id == kind::i8 || in.id == kind::i32)
        return pedantic ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I;
    return pedantic ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
}

struct tolerance {
    long double atol, rtol, input_roundoff, accumulator_roundoff;
};
inline tolerance tolerance_for(kind output, const std::string& mode, int k) {
    tolerance t{2e-6L, 4e-5L, 0, std::min(.1L, static_cast<long double>(k) * 6e-8L)};
    if (output == kind::f64 || output == kind::c128)
        t = {2e-13L, 3e-12L, 0, static_cast<long double>(k) * 1.2e-16L};
    if (output == kind::f16) t.rtol = 8e-4L;
    if (output == kind::bf16) t.rtol = 6e-3L;
    if (output == kind::i32) t = {0, 0, 0, 0};
    if (mode == "tf32" || mode == "fast_f16") t.input_roundoff = 1.1e-3L;
    if (mode == "fast_bf16") t.input_roundoff = 9e-3L;
    if (mode == "f16" || mode == "f16_pedantic")
        t.accumulator_roundoff = std::min(1.L, static_cast<long double>(k) * 5e-4L);
    return t;
}
inline bool compare_value(result& out, wide observed, wide expected, long double sum_abs,
                          tolerance t) {
    const long double err = std::abs(observed - expected);
    const long double rel = err / std::max(1e-30L, std::abs(expected));
    out.max_abs_error = std::max(out.max_abs_error, static_cast<double>(err));
    out.max_rel_error = std::max(out.max_rel_error, static_cast<double>(rel));
    const long double bound = t.atol + t.rtol * std::abs(expected) +
                             (t.input_roundoff + t.accumulator_roundoff) * sum_abs;
    return std::isfinite(static_cast<double>(observed.real())) &&
           std::isfinite(static_cast<double>(observed.imag())) && err <= bound;
}

// ARCHBENCH_HOST_GEMM_DIAGNOSTICS_BEGIN
// This host-only block is compiled independently by its regression test. It
// records evidence only; the numerical acceptance rule remains compare_value.
struct validation_failure_sample {
    uint64_t logical_index = 0;
    int row = 0, column = 0, batch = 0;
    wide observed{}, expected{}, preactivation{};
    long double sum_abs = 0, bound = 0;

    long double absolute_error() const { return std::abs(observed - expected); }
    long double relative_error() const {
        return absolute_error() / std::max(1e-30L, std::abs(expected));
    }
    long double error_over_bound() const {
        const auto error = absolute_error();
        if (!std::isfinite(error) || !std::isfinite(bound) || bound < 0 ||
            !std::isfinite(static_cast<double>(observed.real())) ||
            !std::isfinite(static_cast<double>(observed.imag())))
            return std::numeric_limits<long double>::infinity();
        if (bound == 0)
            return error == 0 ? 0 : std::numeric_limits<long double>::infinity();
        return error / bound;
    }
};

struct validation_failure_details {
    size_t failed_samples = 0;
    validation_failure_sample first{}, worst{};

    void consider(bool passed, const validation_failure_sample& sample) {
        if (passed) return;
        if (failed_samples++ == 0) first = worst = sample;
        else if (sample.error_over_bound() > worst.error_over_bound()) worst = sample;
    }
    static std::string number(long double value) {
        std::ostringstream text;
        text.precision(std::numeric_limits<long double>::max_digits10);
        text << std::scientific << value;
        return text.str();
    }
    static void record_sample(std::map<std::string,std::string>& metadata,
                              const std::string& prefix,
                              const validation_failure_sample& sample) {
        metadata[prefix+"logical_index"] = std::to_string(sample.logical_index);
        metadata[prefix+"row"] = std::to_string(sample.row);
        metadata[prefix+"column"] = std::to_string(sample.column);
        metadata[prefix+"batch"] = std::to_string(sample.batch);
        metadata[prefix+"observed_real"] = number(sample.observed.real());
        metadata[prefix+"observed_imag"] = number(sample.observed.imag());
        metadata[prefix+"expected_real"] = number(sample.expected.real());
        metadata[prefix+"expected_imag"] = number(sample.expected.imag());
        metadata[prefix+"preactivation_real"] = number(sample.preactivation.real());
        metadata[prefix+"preactivation_imag"] = number(sample.preactivation.imag());
        metadata[prefix+"absolute_error"] = number(sample.absolute_error());
        metadata[prefix+"relative_error"] = number(sample.relative_error());
        metadata[prefix+"tolerance_bound"] = number(sample.bound);
        metadata[prefix+"error_over_bound"] = number(sample.error_over_bound());
        metadata[prefix+"bound_sum_abs"] = number(sample.sum_abs);
    }
    void record(std::map<std::string,std::string>& metadata,
                std::map<std::string,double>& metrics) const {
        metrics["validation_failed_samples"] = static_cast<double>(failed_samples);
        if (!failed_samples) return;
        metadata["validation_worst_failure_selection"] =
            "largest absolute_error/tolerance_bound among failed samples; nonfinite values rank first; ties keep first";
        record_sample(metadata,"validation_first_failure.",first);
        record_sample(metadata,"validation_worst_failure.",worst);
    }
};
// ARCHBENCH_HOST_GEMM_DIAGNOSTICS_END

template<class T> __global__ void gather_matrix_kernel(const T* data, const uint64_t* logical,
        double2* values, size_t samples, int m, int n, int ld, size_t stride) {
    for (size_t s = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         s < samples; s += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const uint64_t t = logical[s];
        const uint64_t row = t % m, col = (t / m) % n, batch = t / (uint64_t(m) * n);
        values[s] = as_double2(data[batch * stride + col * ld + row]);
    }
}
inline std::vector<uint64_t> sample_indices(size_t total, size_t maximum) {
    const size_t count = std::min(total, maximum);
    std::vector<uint64_t> indices(count);
    for (size_t i = 0; i < count; ++i)
        indices[i] = count == total ? i : (i < 4 ? (i == 0 ? 0 : total - i) :
                            ((uint64_t(hash_index(i, 0xbadec0deu)) << 32) |
                              hash_index(i, 0x47abc103u)) % total);
    return indices;
}
inline std::vector<double2> gather_matrix(context& ctx, const void* ptr, type_info t,
        const std::vector<uint64_t>& indices, int m, int n, int ld, size_t stride) {
    device_buffer<uint64_t> device_indices(indices.size());
    device_buffer<double2> device_values(indices.size());
    CUDA_CHECK(cudaMemcpyAsync(device_indices.get(), indices.data(), indices.size() * sizeof(uint64_t),
                               cudaMemcpyHostToDevice, ctx.stream));
    const int blocks = static_cast<int>((indices.size() + 255) / 256);
#define GATHER_CASE(k, T) case kind::k: gather_matrix_kernel<T><<<blocks,256,0,ctx.stream>>>(static_cast<const T*>(ptr), device_indices.get(), device_values.get(), indices.size(), m,n,ld,stride); break
    switch (t.id) {
    GATHER_CASE(f16, __half); GATHER_CASE(bf16, __nv_bfloat16); GATHER_CASE(f32, float);
    GATHER_CASE(f64, double); GATHER_CASE(c64, cuComplex); GATHER_CASE(c128, cuDoubleComplex);
    GATHER_CASE(i8, int8_t); GATHER_CASE(i32, int32_t); GATHER_CASE(e4m3, __nv_fp8_e4m3);
    GATHER_CASE(e5m2, __nv_fp8_e5m2);
    default: throw std::invalid_argument("FP4 output validation is not implemented");
    }
#undef GATHER_CASE
    CUDA_CHECK(cudaGetLastError());
    std::vector<double2> values(indices.size());
    CUDA_CHECK(cudaMemcpyAsync(values.data(), device_values.get(), values.size() * sizeof(double2),
                               cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    return values;
}

inline size_t block_scale_count(int outer, int k, int vector_size) {
    return product(product((size_t(outer) + 127) / 128,
                           ((size_t(k) + vector_size - 1) / vector_size + 3) / 4), 512);
}
inline size_t block_scale_offset(int outer, int reduction, int k, int vector_size) {
    return gemm_software::scale_offset(outer,reduction,k,vector_size);
}
inline double scale_at(size_t offset, uint32_t seed) {
    const int exponent = static_cast<int>(hash_index(offset, seed) % 3u) - 1;
    return std::ldexp(1.0, exponent);
}
inline std::vector<uint8_t> make_block_scales(size_t count, uint32_t seed, bool fp4) {
    std::vector<uint8_t> out(count);
    for (size_t i = 0; i < count; ++i) {
        const int exponent = static_cast<int>(hash_index(i, seed) % 3u) - 1;
        out[i] = static_cast<uint8_t>(fp4 ? ((7 + exponent) << 3) : 127 + exponent);
    }
    return out;
}

struct epilogue_config {
    cublasLtEpilogue_t value;
    bool bias, relu, gelu;
};
inline epilogue_config epilogue(const std::string& s) {
    if (s == "default") return {CUBLASLT_EPILOGUE_DEFAULT, false, false, false};
    if (s == "bias") return {CUBLASLT_EPILOGUE_BIAS, true, false, false};
    if (s == "relu") return {CUBLASLT_EPILOGUE_RELU, false, true, false};
    if (s == "gelu") return {CUBLASLT_EPILOGUE_GELU, false, false, true};
    if (s == "bias_relu") return {CUBLASLT_EPILOGUE_RELU_BIAS, true, true, false};
    if (s == "bias_gelu") return {CUBLASLT_EPILOGUE_GELU_BIAS, true, false, true};
    throw std::invalid_argument("unknown epilogue: " + s);
}

inline void validate_gemm(context& ctx, result& out, const problem& p, type_info a, type_info b,
        type_info c, type_info bias_type, const scalar_pair& scalar, const void* d,
        epilogue_config epi, int block_size, double scale_a, double scale_b,
        const std::string& validation_compute="") {
    const size_t total = product(product(p.m, p.n), p.batch);
    out.metadata["input_pattern"]=ctx.opts.get("input-pattern","seeded");
    if(rank1_problem(ctx,p,a,b,c)) {
        // Bound host memory while checking every logical output, including
        // non-square shapes and leading-dimension padding. Copies are untimed.
        const size_t chunk_elements=size_t(1)<<20;
        std::vector<float> host(std::min(chunk_elements,p.count_c));
        bool good=true;validation_failure_details failures;
        for(size_t offset=0;offset<p.count_c;offset+=host.size()) {
            const size_t count=std::min(host.size(),p.count_c-offset);
            CUDA_CHECK(cudaMemcpy(host.data(),static_cast<const float*>(d)+offset,count*sizeof(float),cudaMemcpyDeviceToHost));
            for(size_t j=0;j<count;++j) {
                const size_t physical=offset+j;
                const int row=int(physical%p.ldc),column=int(physical/p.ldc);
                if(row>=p.m||column>=p.n)continue;
                const long double expected=gemm_software::rank1_expected(row,column,p.k);
                const wide observed{host[j],0};
                const bool passed=compare_value(out,observed,wide{expected,0},std::abs(expected),{0,0,0,0});
                if(!passed)failures.consider(false,{size_t(column)*p.m+row,row,column,0,observed,
                    {expected,0},{expected,0},std::abs(expected),0});
                good=good&&passed;
            }
        }
        out.validation=good?"passed":"failed";
        if(!good) {out.status="validation_failed";out.reason="rank1 GPU output differs from its exact binary-fraction analytic reference";}
        out.metadata["reference"]="analytic exact FP32: K*((row%13-6)/8)*((column%11-5)/8)";
        out.metadata["validation_scope"]="all_output_elements";
        out.metrics["validation_elements"]=double(total);
        out.metrics["reference_atol"]=0;out.metrics["reference_rtol"]=0;
        out.metrics["reference_input_roundoff"]=0;out.metrics["reference_accumulator_roundoff"]=0;
        failures.record(out.metadata,out.metrics);return;
    }
    const size_t limit = static_cast<size_t>(positive_int(ctx.opts, "validation-samples", 128));
    const auto indices = sample_indices(total, limit);
    const auto observed = gather_matrix(ctx, d, c, indices, p.m, p.n, p.ldc, p.stride_c);
    auto tolerance = tolerance_for(c.id,validation_compute.empty()?ctx.opts.get("compute","default"):validation_compute,p.k);
    validation_failure_details failures;
    bool good = true;
    for (size_t s = 0; s < indices.size(); ++s) {
        const uint64_t t = indices[s];
        const int row = static_cast<int>(t % p.m), col = static_cast<int>((t / p.m) % p.n);
        const int batch = static_cast<int>(t / (uint64_t(p.m) * p.n));
        wide sum{};
        long double sum_abs = 0;
        for (int reduction = 0; reduction < p.k; ++reduction) {
            auto av = generated_wide(a.id, p.ai(row, reduction, batch), 17);
            auto bv = generated_wide(b.id, p.bi(reduction, col, batch), 29);
            if (p.ta == CUBLAS_OP_C) av = std::conj(av);
            if (p.tb == CUBLAS_OP_C) bv = std::conj(bv);
            if (block_size) {
                av *= scale_at(block_scale_offset(row, reduction, p.k, block_size), 41);
                bv *= scale_at(block_scale_offset(col, reduction, p.k, block_size), 53);
            } else { av *= scale_a; bv *= scale_b; }
            sum += av * bv;
            sum_abs += std::abs(av) * std::abs(bv);
        }
        const auto cv = generated_wide(c.id, batch * p.stride_c + col * size_t(p.ldc) + row, 71);
        wide expected = scalar.alpha * sum + scalar.beta * cv;
        sum_abs = std::abs(scalar.alpha) * sum_abs + std::abs(scalar.beta * cv);
        if (epi.bias) expected += generated_wide(bias_type.id, row, 97);
        const wide preactivation = expected;
        if (epi.relu) expected = std::max(0.L, expected.real());
        if (epi.gelu) {
            const long double x = expected.real();
            expected = .5L * x * (1 + std::tanh(.79788456080286535588L * (x + .044715L * x * x * x)));
            // Account for the documented approximate nonlinear epilogue.
            tolerance.atol = std::max(tolerance.atol, 3e-5L);
        }
        const wide observed_value{observed[s].x, observed[s].y};
        const bool passed = compare_value(out, observed_value, expected, sum_abs, tolerance);
        if (!passed) {
            // Keep the exact acceptance bound above, including the pre-epilogue
            // absolute product sum. This diagnostic does not relax that bound.
            const long double bound = tolerance.atol + tolerance.rtol * std::abs(expected) +
                (tolerance.input_roundoff + tolerance.accumulator_roundoff) * sum_abs;
            failures.consider(false,{t,row,col,batch,observed_value,expected,preactivation,sum_abs,bound});
        }
        good = passed && good;
    }
    out.validation = good ? "passed" : "failed";
    if (!good) { out.status = "validation_failed"; out.reason = "GPU output exceeds the documented condition-aware reference bound"; }
    out.metadata["reference"] = "CPU long-double dot products of actual stored/quantized inputs";
    out.metadata["validation_scope"] = indices.size() == total ? "all_output_elements" : "deterministic_output_samples";
    out.metrics["validation_elements"] = static_cast<double>(indices.size());
    out.metrics["reference_atol"] = static_cast<double>(tolerance.atol);
    out.metrics["reference_rtol"] = static_cast<double>(tolerance.rtol);
    out.metrics["reference_input_roundoff"] = static_cast<double>(tolerance.input_roundoff);
    out.metrics["reference_accumulator_roundoff"] = static_cast<double>(tolerance.accumulator_roundoff);
    failures.record(out.metadata,out.metrics);
}

inline void record_problem(result& out, const problem& p, type_info a, type_info b, type_info c, bool read_c) {
    out.execution = "library_selected";
    out.unit = "Gop/s";
    out.work_per_iteration = (a.complex ? 8.0 : 2.0) * p.m * p.n * p.k * p.batch;
    out.bytes_per_iteration = p.batch * (static_cast<double>(p.m) * p.k * a.bits / 8.0 +
                                        static_cast<double>(p.k) * p.n * b.bits / 8.0 +
                                        static_cast<double>(p.m) * p.n * c.bits / 8.0 * (read_c ? 2 : 1));
    out.metadata["operation_count"] = a.complex ? "8 real operations per complex MAC" : "2 operations per real/integer MAC";
    out.metadata["byte_count"] = "useful tensor bytes for one read of A/B and write of D plus C only if beta!=0; excludes padding, scales, workspace, and cache reuse; not measured DRAM bytes";
    out.metadata["layout"] = "column_major";
    out.metadata["execution_evidence"] = "API acceptance only; inspect Nsight/SASS to identify the selected hardware path";
    out.metrics["lda"] = p.lda; out.metrics["ldb"] = p.ldb; out.metrics["ldc"] = p.ldc;
    out.metrics["stride_a"] = static_cast<double>(p.stride_a);
    out.metrics["stride_b"] = static_cast<double>(p.stride_b);
    out.metrics["stride_c"] = static_cast<double>(p.stride_c);
}

inline cublasGemmAlgo_t classic_gemm_algorithm(const options& opts) {
    const auto value=opts.get("cublas-algo","default");
    if(value=="default")return CUBLAS_GEMM_DEFAULT;
    if(value=="default_tensor_op")return CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    throw std::invalid_argument("cublas-algo must be default or default_tensor_op");
}

#include "gemm_software.inc"

inline void record_fixedpoint_indicator(context& ctx,result& out,const device_buffer<int>& indicator,
                                        const std::string& phase) {
    int raw=-1;
    CUDA_CHECK(cudaMemcpyAsync(&raw,indicator.get(),sizeof(raw),cudaMemcpyDeviceToHost,ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    out.metrics["fixedpoint_indicator_"+phase+"_raw"]=raw;
    out.metrics["fixedpoint_indicator_raw"]=raw;
    // Retain the historical field for report compatibility; its name must not
    // turn a library-written integer (4247 in supplied runs) into decoded bits.
    out.metrics["fixedpoint_reported_mantissa_bits"]=raw;
    out.metadata["fixedpoint_used_last_call"]=raw>0?"true":raw==-1?"false":"unknown";
    out.metadata["fixedpoint_indicator_flag_basis"]="documented sign convention only: -1 reports no emulation, positive reports emulation; not instruction evidence";
    out.metadata["fixedpoint_indicator_interpretation"]="raw device int; no encoding decoded and no literal precision inferred; 4247 observed in supplied CUDA13.5 runs remains unexplained";
    out.metadata["fixedpoint_indicator_last_phase"]=phase;
    out.metadata["fixedpoint_precision_decoded"]="false";
    out.metadata["fixedpoint_reported_mantissa_bits_legacy_name"]="compatibility alias for fixedpoint_indicator_raw; not a decoded bit count";
}

inline void cublas_gemm(context& ctx, result& out) {
    const auto start = clock_type::now();
    const problem p(ctx.opts);
    const auto algorithm=classic_gemm_algorithm(ctx.opts);
    out.metadata["cublas_algorithm_control"]=ctx.opts.get("cublas-algo","default");
    out.metadata["cublas_algorithm_control_application"]="requested cublasGemmAlgo_t in GemmEx/StridedBatchedEx/BatchedEx; not proof of selected hardware instructions";
    out.metrics["requested_cublas_algorithm_enum"]=static_cast<int>(algorithm);
    const std::string dtype = ctx.opts.get("dtype", "f32");
    const auto a = parse_type(dtype), b = parse_type(ctx.opts.get("dtype-b", dtype));
    const std::string output_name = ctx.opts.get("output-dtype", dtype == "i8" ? "i32" : dtype);
    const auto c = parse_type(output_name);
    if (a.narrow || b.narrow || c.narrow) {
        skip(out, "FP8/FP4 GEMM uses the cuBLASLt backend; classic GemmEx does not expose this path"); return;
    }
    if (ctx.opts.get("epilogue", "default") != "default") {
        skip(out, "fused epilogues require --backend cublaslt"); return;
    }
    if(ctx.opts.get("scale-mode","tensor")!="tensor"||ctx.opts.real("scale-a",1)!=1||ctx.opts.real("scale-b",1)!=1||ctx.opts.flag("fast-accum",false)) {
        skip(out,"classic GemmEx does not implement narrow-input scales or the cuBLASLt fast-accum descriptor; use cublaslt or an explicit software reference");return;
    }
    const auto compute = compute_type(ctx.opts, a);
    scalar_pair scalar(ctx.opts, c, compute);
    if (c.id == kind::i32 && (9.L * p.k * std::abs(scalar.alpha) + 3.L * std::abs(scalar.beta) > INT32_MAX))
        throw std::invalid_argument("integer reference would exceed INT32; reduce K or alpha/beta");
    const size_t ab = storage_bytes(p.count_a,a), bb = storage_bytes(p.count_b,b), cb = storage_bytes(p.count_c,c);
    const auto workspace_mib = ctx.opts.integer("workspace-mib", 32);
    if (workspace_mib < 0) throw std::invalid_argument("workspace-mib must be nonnegative");
    const size_t workspace_bytes = product(static_cast<size_t>(workspace_mib), size_t(1) << 20);
    const bool restore_c = std::abs(scalar.beta) != 0;
    const auto batch_mode = ctx.opts.get("batch-mode", "strided");
    if(batch_mode!="strided" && batch_mode!="pointer")
        throw std::invalid_argument("batch-mode must be strided or pointer");
    const bool pointer_batch = batch_mode == "pointer";
    ctx.memory_guard(sum_sizes({ab,bb,product(cb,restore_c?2:1),workspace_bytes,
                               product(product(3,size_t(p.batch)),sizeof(void*)),sizeof(int)}));
    device_buffer<uint8_t> da(ab), db(bb), dc(cb), saved_c(restore_c ? cb : 1), workspace(std::max<size_t>(workspace_bytes,1));
    device_buffer<int> emulation_bits(1);
    CUDA_CHECK(cudaMemsetAsync(emulation_bits.get(),0xff,sizeof(int),ctx.stream));
    device_buffer<const void*> pointer_a(pointer_batch?p.batch:0), pointer_b(pointer_batch?p.batch:0);
    device_buffer<void*> pointer_c(pointer_batch?p.batch:0);
    if(pointer_batch) {
        std::vector<const void*> ha(p.batch),hb(p.batch);std::vector<void*> hc(p.batch);
        for(int i=0;i<p.batch;++i) {
            ha[i]=da.get()+storage_bytes(i*p.stride_a,a);
            hb[i]=db.get()+storage_bytes(i*p.stride_b,b);
            hc[i]=dc.get()+storage_bytes(i*p.stride_c,c);
        }
        CUDA_CHECK(cudaMemcpyAsync(pointer_a.get(),ha.data(),ha.size()*sizeof(void*),cudaMemcpyHostToDevice,ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(pointer_b.get(),hb.data(),hb.size()*sizeof(void*),cudaMemcpyHostToDevice,ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(pointer_c.get(),hc.data(),hc.size()*sizeof(void*),cudaMemcpyHostToDevice,ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    }
    fill_gemm_inputs(ctx,p,a,b,c,da.get(),db.get(),dc.get());
    if (restore_c) CUDA_CHECK(cudaMemcpyAsync(saved_c.get(), dc.get(), cb, cudaMemcpyDeviceToDevice, ctx.stream));
    handle h(ctx.stream);
    if (workspace_bytes) BLAS_CHECK(cublasSetWorkspace(h.value, workspace.get(), workspace_bytes));
    BLAS_CHECK(cublasSetAtomicsMode(h.value, ctx.opts.flag("allow-atomics",false) ? CUBLAS_ATOMICS_ALLOWED : CUBLAS_ATOMICS_NOT_ALLOWED));
    if (ctx.opts.flag("no-reduced-reduction", false))
        BLAS_CHECK(cublasSetMathMode(h.value, static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    const auto sm_count = bounded_int(ctx.opts,"sm-count",0,0,ctx.prop.multiProcessorCount);
    if (sm_count) BLAS_CHECK(cublasSetSmCountTarget(h.value,sm_count));
#if CUDART_VERSION >= 13050
    BLAS_CHECK(cublasSetEmulationStrategy(h.value,emulation_strategy(ctx.opts)));
    out.metadata["emulation_strategy"] = ctx.opts.get("emulation-strategy","eager");
    if(ctx.opts.get("compute")=="fixedpoint") {
        BLAS_CHECK(cublasSetFixedPointEmulationMantissaControl(h.value,mantissa_control(ctx.opts)));
        BLAS_CHECK(cublasSetFixedPointEmulationMaxMantissaBitCount(h.value,bounded_int(ctx.opts,"mantissa-bits",0,0,INT32_MAX)));
        BLAS_CHECK(cublasSetFixedPointEmulationMantissaBitOffset(h.value,bounded_int(ctx.opts,"mantissa-offset",0,INT32_MIN,INT32_MAX)));
        BLAS_CHECK(cublasSetFixedPointEmulationMantissaBitCountPointer(h.value,emulation_bits.get()));
        int configured_max=0;
        BLAS_CHECK(cublasGetFixedPointEmulationMaxMantissaBitCount(h.value,&configured_max));
        out.metrics["fixedpoint_configured_max_mantissa_bits"]=configured_max;
    }
#endif
    int version = 0; BLAS_CHECK(cublasGetVersion(h.value, &version));
    out.metrics["cublas_version"] = version;
    const auto launch = [&] {
        if(pointer_batch) {
            BLAS_CHECK(cublasGemmBatchedEx(h.value,p.ta,p.tb,p.m,p.n,p.k,scalar.a,
                pointer_a.get(),a.cuda,p.lda,pointer_b.get(),b.cuda,p.ldb,scalar.b,
                pointer_c.get(),c.cuda,p.ldc,p.batch,compute,algorithm));
        } else if (p.batch == 1) {
            BLAS_CHECK(cublasGemmEx(h.value,p.ta,p.tb,p.m,p.n,p.k,scalar.a,
                da.get(),a.cuda,p.lda,db.get(),b.cuda,p.ldb,scalar.b,dc.get(),c.cuda,p.ldc,
                compute,algorithm));
        } else {
            BLAS_CHECK(cublasGemmStridedBatchedEx(h.value,p.ta,p.tb,p.m,p.n,p.k,scalar.a,
                da.get(),a.cuda,p.lda,static_cast<long long>(p.stride_a),
                db.get(),b.cuda,p.ldb,static_cast<long long>(p.stride_b),scalar.b,
                dc.get(),c.cuda,p.ldc,static_cast<long long>(p.stride_c),p.batch,compute,algorithm));
        }
    };
    const auto restore = [&] { if (restore_c) CUDA_CHECK(cudaMemcpyAsync(dc.get(),saved_c.get(),cb,cudaMemcpyDeviceToDevice,ctx.stream)); };
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    out.metrics["setup_ms"] = std::chrono::duration<double,std::milli>(clock_type::now()-start).count();
    restore(); launch(); CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    record_problem(out,p,a,b,c,restore_c);
    out.metadata["batch_mode"]=batch_mode;
    out.metadata["timing"] = restore_c ? "individual_cuda_events_excluding_C_reset" : "cuda_event_batch";
    validate_gemm(ctx,out,p,a,b,c,c,scalar,dc.get(),epilogue("default"),0,1,1);
    if(ctx.opts.get("compute")=="fixedpoint")record_fixedpoint_indicator(ctx,out,emulation_bits,"validation");
    if (out.validation != "passed") return;
    out.samples_ms = restore_c ? measure_restored(ctx,restore,launch) : ctx.measure(launch);
    if(ctx.opts.get("compute")=="fixedpoint")record_fixedpoint_indicator(ctx,out,emulation_bits,"timing");
    validate_gemm(ctx,out,p,a,b,c,c,scalar,dc.get(),epilogue("default"),0,1,1);
    out.metadata["validation_timing_phase"]="before and after performance measurement";
}

inline void lt_gemm(context& ctx, result& out) {
    const auto start = clock_type::now();
    const problem p(ctx.opts);
    if(classic_gemm_algorithm(ctx.opts)!=CUBLAS_GEMM_DEFAULT) {
        skip(out,"cublas-algo is a classic-GemmEx control; cuBLASLt selects its own validated heuristic algorithms");return;
    }
    const std::string dtype = ctx.opts.get("dtype", "f32");
    const auto a = parse_type(dtype), b = parse_type(ctx.opts.get("dtype-b", dtype));
    const std::string default_output = a.narrow ? "f16" : dtype == "i8" ? "i32" : dtype;
    const auto c = parse_type(ctx.opts.get("output-dtype", default_output));
    if(ctx.opts.flag("no-reduced-reduction",false)||ctx.opts.flag("allow-atomics",false)||ctx.opts.integer("sm-count",0)!=0) {
        skip(out,"no-reduced-reduction, allow-atomics, and sm-count are classic-cuBLAS handle controls; this Lt backend does not apply them");return;
    }
    if(ctx.opts.get("batch-mode","strided")!="strided") {
        skip(out,"pointer-array batching is implemented in classic cuBLAS; this Lt backend uses strided layouts");return;
    }
    if (c.narrow) { skip(out,"narrow output quantization is separate from this GEMM benchmark; choose f16/bf16/f32 output"); return; }
    const auto compute = compute_type(ctx.opts, a);
    scalar_pair scalar(ctx.opts,c,compute);
    const auto epi = epilogue(ctx.opts.get("epilogue","default"));
    const bool fp4 = a.id == kind::e2m1 || b.id == kind::e2m1;
    const auto scale_mode = ctx.opts.get("scale-mode", fp4 ? "block16" : "tensor");
    const int block_size = scale_mode == "block16" ? 16 : scale_mode == "block32" ? 32 : 0;
    if (scale_mode != "tensor" && scale_mode != "block16" && scale_mode != "block32")
        throw std::invalid_argument("scale-mode must be tensor, block16, or block32");
    if (fp4 && (a.id != kind::e2m1 || b.id != kind::e2m1 || block_size != 16)) {
        skip(out,"cuBLASLt FP4 requires FP4 on both operands with block16 UE4M3 scales"); return;
    }
    if (block_size && (!a.narrow || !b.narrow || p.batch != 1 || p.ta != CUBLAS_OP_T || p.tb != CUBLAS_OP_N)) {
        skip(out,"this block-scaled backend requires narrow A/B, TN layout, and batch=1"); return;
    }
    if (block_size == 16 && !fp4) { skip(out,"block16 UE4M3 scaling is implemented for NVFP4 only"); return; }
    if (block_size == 32 && (a.id == kind::e2m1 || b.id == kind::e2m1)) { skip(out,"MXFP4 is not a cuBLASLt API path in this module"); return; }
    const float scale_a = static_cast<float>(ctx.opts.real("scale-a",1));
    const float scale_b = static_cast<float>(ctx.opts.real("scale-b",1));
    if (!std::isfinite(scale_a) || !std::isfinite(scale_b) || scale_a <= 0 || scale_b <= 0)
        throw std::invalid_argument("tensor scales must be finite and positive");
    if (block_size && (scale_a != 1 || scale_b != 1))
        throw std::invalid_argument("scale-a/scale-b are tensor-scale parameters; block scale values are generated independently");
    const type_info bias_type = a.narrow && c.id == kind::f32 ? parse_type("bf16") : c;
    const size_t ab=storage_bytes(p.count_a,a), bb=storage_bytes(p.count_b,b), cb=storage_bytes(p.count_c,c);
    const auto workspace_mib=ctx.opts.integer("workspace-mib",32);
    if(workspace_mib<0) throw std::invalid_argument("workspace-mib must be nonnegative");
    const size_t workspace_bytes=product(static_cast<size_t>(workspace_mib),size_t(1)<<20);
    const size_t sa_count=block_size ? block_scale_count(p.m,p.k,block_size):sizeof(float);
    const size_t sb_count=block_size ? block_scale_count(p.n,p.k,block_size):sizeof(float);
    const size_t bias_bytes=storage_bytes(p.m,bias_type);
    ctx.memory_guard(sum_sizes({ab,bb,product(2,cb),workspace_bytes,sa_count,sb_count,bias_bytes,sizeof(int)}));
    device_buffer<uint8_t> da(ab),db(bb),dc(cb),dd(cb),bias(std::max<size_t>(bias_bytes,1));
    device_buffer<uint8_t> sa(sa_count),sb(sb_count),workspace(std::max<size_t>(workspace_bytes,1));
    device_buffer<int> emulation_bits(1);
    CUDA_CHECK(cudaMemsetAsync(emulation_bits.get(),0xff,sizeof(int),ctx.stream));
    fill_gemm_inputs(ctx,p,a,b,c,da.get(),db.get(),dc.get());
    if(epi.bias) fill_buffer(ctx,bias.get(),bias_type,p.m,97);
    if(block_size) {
        const auto hsa=make_block_scales(sa_count,41,fp4), hsb=make_block_scales(sb_count,53,fp4);
        CUDA_CHECK(cudaMemcpyAsync(sa.get(),hsa.data(),hsa.size(),cudaMemcpyHostToDevice,ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(sb.get(),hsb.data(),hsb.size(),cudaMemcpyHostToDevice,ctx.stream));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(sa.get(),&scale_a,sizeof(float),cudaMemcpyHostToDevice,ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(sb.get(),&scale_b,sizeof(float),cudaMemcpyHostToDevice,ctx.stream));
    }
    lt_handle h;
#if CUDART_VERSION >= 13050
    lt_emulation emulation;
#endif
    lt_descriptor desc(compute,scalar.type);
    desc.set(CUBLASLT_MATMUL_DESC_TRANSA,p.ta); desc.set(CUBLASLT_MATMUL_DESC_TRANSB,p.tb);
    desc.set(CUBLASLT_MATMUL_DESC_EPILOGUE,epi.value);
#if CUDART_VERSION >= 13050
    if(ctx.opts.get("compute")=="fixedpoint"||ctx.opts.get("compute")=="bf16x9") {
        emulation.set(CUBLASLT_EMULATION_DESC_STRATEGY,emulation_strategy(ctx.opts));
        if(ctx.opts.get("compute")=="fixedpoint") {
            emulation.set(CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL,mantissa_control(ctx.opts));
            const int bits=bounded_int(ctx.opts,"mantissa-bits",0,0,INT32_MAX);
            const int offset=bounded_int(ctx.opts,"mantissa-offset",0,INT32_MIN,INT32_MAX);
            emulation.set(CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,bits);
            emulation.set(CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET,offset);
            int* pointer=emulation_bits.get();
            emulation.set(CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER,pointer);
        }
        desc.set(CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,emulation.value);
        out.metadata["emulation_strategy"]=ctx.opts.get("emulation-strategy","eager");
    }
#endif
    if(a.narrow && !fp4) {
        const int8_t fast_accum=ctx.opts.flag("fast-accum",false)?1:0;
        desc.set(CUBLASLT_MATMUL_DESC_FAST_ACCUM,fast_accum);
        out.metadata["fast_accum"]=fast_accum?"true":"false";
    }
    if(epi.bias) { void* pointer=bias.get(); desc.set(CUBLASLT_MATMUL_DESC_BIAS_POINTER,pointer); desc.set(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,bias_type.cuda); }
    if(a.narrow || b.narrow) {
        void* pa=sa.get(); void* pb=sb.get();
        desc.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,pa);
        desc.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,pb);
#if CUDART_VERSION >= 12080
        const cublasLtMatmulMatrixScale_t mode = block_size == 16 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 :
                    block_size == 32 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 : CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        desc.set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE,mode); desc.set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE,mode);
#else
        if(block_size) { skip(out,"block-scaled cuBLASLt requires newer toolkit headers"); return; }
#endif
    } else if(scale_a!=1 || scale_b!=1) throw std::invalid_argument("A/B scales apply to narrow-input matmul only");
    lt_layout la(a.cuda,p.ta==CUBLAS_OP_N?p.m:p.k,p.ta==CUBLAS_OP_N?p.k:p.m,p.lda);
    lt_layout lb(b.cuda,p.tb==CUBLAS_OP_N?p.k:p.n,p.tb==CUBLAS_OP_N?p.n:p.k,p.ldb);
    lt_layout lc(c.cuda,p.m,p.n,p.ldc),ld(c.cuda,p.m,p.n,p.ldc);
    if(p.batch>1) {
        la.set(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,p.batch); lb.set(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,p.batch);
        lc.set(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,p.batch); ld.set(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,p.batch);
        const int64_t stride_a=p.stride_a,stride_b=p.stride_b,stride_c=p.stride_c;
        la.set(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,stride_a); lb.set(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,stride_b);
        lc.set(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,stride_c); ld.set(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,stride_c);
    }
    lt_preference pref;
    BLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref.value,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)));
    const int requested=std::min(positive_int(ctx.opts,"algorithms",16),128);
    std::vector<cublasLtMatmulHeuristicResult_t> choices(requested);
    int returned=0;
    BLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(h.value,desc.value,la.value,lb.value,lc.value,ld.value,pref.value,
                                           requested,choices.data(),&returned));
    out.metrics["heuristic_candidates"]=returned;
    if(!returned) { skip(out,"cuBLASLt returned no algorithm for this exact descriptor, workspace, and device"); return; }
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    out.metrics["setup_ms"]=std::chrono::duration<double,std::milli>(clock_type::now()-start).count();
    const auto tune_start=clock_type::now();
    auto raw_launch=[&](const cublasLtMatmulAlgo_t* algorithm) {
        return cublasLtMatmul(h.value,desc.value,scalar.a,da.get(),la.value,db.get(),lb.value,scalar.b,
                             dc.get(),lc.value,dd.get(),ld.value,algorithm,workspace_bytes?workspace.get():nullptr,workspace_bytes,ctx.stream);
    };
    const int forced_index=bounded_int(ctx.opts,"heuristic-index",-1,-1,127);
    const int tune_warmup=bounded_int(ctx.opts,"tune-warmup",3,0,10000);
    const int tune_samples=bounded_int(ctx.opts,"tune-samples",5,1,1000);
    const int tune_repeats=bounded_int(ctx.opts,"tune-repeats",3,1,1000000);
    const int tune_max_repeats=bounded_int(ctx.opts,"tune-max-repeats",256,tune_repeats,1000000);
    const double tune_target_ms=ctx.opts.real("tune-target-ms",1);
    if(tune_target_ms<=0||tune_target_ms>60000)
        throw std::invalid_argument("tune-target-ms must be in (0, 60000]");
    if(forced_index>=returned) {
        skip(out,"heuristic-index is outside the candidates returned for this exact descriptor; increase algorithms or select an available local index");return;
    }
    out.metadata["autotune_selection"]=forced_index<0?"fastest validated repeated-sample median":"explicit descriptor-local heuristic index; correctness is mandatory";
    out.metadata["autotune_scope"]="sampled reference validation before warmup/calibration/repeated timing; failed candidates cannot be selected; independent final measurement";
    out.metrics["tune_warmup"]=tune_warmup;out.metrics["tune_samples"]=tune_samples;
    out.metrics["tune_min_repeats"]=tune_repeats;out.metrics["tune_max_repeats"]=tune_max_repeats;
    out.metrics["tune_target_ms"]=tune_target_ms;
    std::vector<gemm_software::candidate_score> scores;
    result first_numerical_failure;
    int numerical_failures=0,validated_candidates=0,timed_candidates=0;
    std::ostringstream timings;
    event_pair events;
    auto record_config=[&](int index,const std::string& prefix) {
        const auto read=[&](cublasLtMatmulAlgoConfigAttributes_t attr,const char* key) {
            int value=0;size_t written=0;
            if(cublasLtMatmulAlgoConfigGetAttribute(&choices[index].algo,attr,&value,sizeof(value),&written)==CUBLAS_STATUS_SUCCESS&&written==sizeof(value))
                out.metrics[prefix+key]=value;
        };
        read(CUBLASLT_ALGO_CONFIG_ID,"algorithm_id");read(CUBLASLT_ALGO_CONFIG_TILE_ID,"tile_id");
        read(CUBLASLT_ALGO_CONFIG_SPLITK_NUM,"split_k");read(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,"reduction_scheme");
        read(CUBLASLT_ALGO_CONFIG_STAGES_ID,"stages");
    };
    const auto is_descriptor_rejection=[](cublasStatus_t status) {
        return status==CUBLAS_STATUS_NOT_SUPPORTED||status==CUBLAS_STATUS_INVALID_VALUE||status==CUBLAS_STATUS_ARCH_MISMATCH;
    };
    for(int i=0;i<returned;++i) {
        const std::string prefix="candidate."+std::to_string(i)+".";
        record_config(i,prefix);
        out.metrics[prefix+"workspace_bytes"]=static_cast<double>(choices[i].workspaceSize);
        out.metrics[prefix+"heuristic_status"]=static_cast<int>(choices[i].state);
        if(forced_index>=0&&i!=forced_index) {out.metadata[prefix+"status"]="not_requested";continue;}
        if(choices[i].state!=CUBLAS_STATUS_SUCCESS||choices[i].workspaceSize>workspace_bytes) {
            out.metadata[prefix+"status"]="heuristic_rejected";continue;
        }
        auto elapsed=[&](int repetitions) {
            CUDA_CHECK(cudaEventRecord(events.begin,ctx.stream));
            cublasStatus_t status=CUBLAS_STATUS_SUCCESS;
            for(int r=0;r<repetitions;++r) {
                status=raw_launch(&choices[i].algo);
                if(status!=CUBLAS_STATUS_SUCCESS)break;
            }
            CUDA_CHECK(cudaEventRecord(events.end,ctx.stream));
            CUDA_CHECK(cudaEventSynchronize(events.end));
            float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,events.begin,events.end));
            if(status!=CUBLAS_STATUS_SUCCESS&&!is_descriptor_rejection(status))BLAS_CHECK(status);
            return std::make_pair(status,static_cast<double>(ms));
        };
        const auto validation_call=elapsed(1);
        out.metrics[prefix+"validation_launch_status"]=static_cast<int>(validation_call.first);
        out.metrics[prefix+"validation_call_ms"]=validation_call.second;
        if(validation_call.first!=CUBLAS_STATUS_SUCCESS) {
            out.metadata[prefix+"status"]="launch_rejected";continue;
        }
        result candidate_validation;
        const auto validation_start=clock_type::now();
        validate_gemm(ctx,candidate_validation,p,a,b,c,bias_type,scalar,dd.get(),epi,block_size,scale_a,scale_b);
        out.metrics[prefix+"validation_host_ms"]=std::chrono::duration<double,std::milli>(clock_type::now()-validation_start).count();
        out.metadata[prefix+"validation"]=candidate_validation.validation;
        out.metrics[prefix+"max_abs_error"]=candidate_validation.max_abs_error;
        out.metrics[prefix+"max_rel_error"]=candidate_validation.max_rel_error;
        for(const auto& item:candidate_validation.metadata)out.metadata[prefix+item.first]=item.second;
        for(const auto& item:candidate_validation.metrics)out.metrics[prefix+item.first]=item.second;
        if(candidate_validation.validation!="passed") {
            out.metadata[prefix+"status"]="validation_failed";
            out.metadata[prefix+"performance_timing"]="excluded; validation-call time is diagnostic only";
            if(numerical_failures++==0)first_numerical_failure=candidate_validation;
            scores.push_back({i,false,{}});continue;
        }
        ++validated_candidates;
        cublasStatus_t warmup_status=CUBLAS_STATUS_SUCCESS;
        const auto warmup_start=clock_type::now();
        for(int r=0;r<tune_warmup;++r) {
            warmup_status=raw_launch(&choices[i].algo);
            if(warmup_status!=CUBLAS_STATUS_SUCCESS)break;
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
        out.metrics[prefix+"warmup_host_ms"]=std::chrono::duration<double,std::milli>(clock_type::now()-warmup_start).count();
        if(warmup_status!=CUBLAS_STATUS_SUCCESS) {
            if(!is_descriptor_rejection(warmup_status))BLAS_CHECK(warmup_status);
            out.metadata[prefix+"status"]="warmup_rejected";continue;
        }
        const auto calibration=elapsed(tune_repeats);
        out.metrics[prefix+"calibration_ms"]=calibration.second;
        if(calibration.first!=CUBLAS_STATUS_SUCCESS) {
            out.metadata[prefix+"status"]="calibration_rejected";continue;
        }
        const int repetitions=std::clamp(int(std::min<double>(tune_max_repeats,
            std::ceil(tune_target_ms/std::max(1e-6,calibration.second/tune_repeats)))),tune_repeats,tune_max_repeats);
        gemm_software::candidate_score score{i,true,{}};
        std::ostringstream raw_samples;raw_samples.precision(17);
        for(int s=0;s<tune_samples;++s) {
            const auto interval=elapsed(repetitions);
            if(interval.first!=CUBLAS_STATUS_SUCCESS) {score.validated=false;break;}
            const double per_call=interval.second/repetitions;
            score.samples_ms.push_back(per_call);
            if(s)raw_samples<<",";
            raw_samples<<per_call;
        }
        out.metrics[prefix+"timing_repeats"]=repetitions;
        out.metadata[prefix+"samples_ms"]=raw_samples.str();
        const double median=gemm_software::candidate_median(score);
        if(std::isfinite(median)) {
            out.metrics[prefix+"median_ms"]=median;
            out.metadata[prefix+"status"]="validated_and_timed";
            if(timings.tellp()>0)timings<<",";
            timings.precision(17);timings<<i<<":"<<median;++timed_candidates;
        } else out.metadata[prefix+"status"]="timing_rejected";
        scores.push_back(std::move(score));
    }
    const int selected=gemm_software::fastest_validated(scores);
    out.metrics["autotune_ms"]=std::chrono::duration<double,std::milli>(clock_type::now()-tune_start).count();
    out.metrics["autotune_numerical_rejections"]=numerical_failures;
    out.metrics["autotune_validated_candidates"]=validated_candidates;
    out.metrics["autotune_timed_candidates"]=timed_candidates;
    out.metadata["candidate_ms"]=timings.str();
    out.metrics["cublaslt_version"]=static_cast<double>(cublasLtGetVersion());
    record_problem(out,p,a,b,c,std::abs(scalar.beta)!=0);
    if(selected<0) {
        if(numerical_failures) {
            out.status="validation_failed";out.validation="failed";
            out.reason="No requested cuBLASLt heuristic passed both sampled correctness validation and valid timing; numerical rejection evidence is retained per candidate";
            out.max_abs_error=first_numerical_failure.max_abs_error;out.max_rel_error=first_numerical_failure.max_rel_error;
            for(const auto& item:first_numerical_failure.metadata)out.metadata[item.first]=item.second;
            for(const auto& item:first_numerical_failure.metrics)out.metrics[item.first]=item.second;
            out.metadata["validation_summary_source"]="first numerically rejected candidate; see candidate.* for every rejection";
        } else skip(out,"all requested heuristic candidates rejected the descriptor or produced invalid timing");
        return;
    }
    // Identity is stored before the independent final validation as well.
    if(block_size) out.metrics["scale_storage_bytes"]=static_cast<double>(sa_count+sb_count);
    out.metrics["selected_heuristic_index"]=selected;
    out.metrics["selected_workspace_bytes"]=static_cast<double>(choices[selected].workspaceSize);
    out.metadata["scale_mode"]=scale_mode;
    out.metadata["timing"]="cuda_event_batch; separate immutable C and D";
    record_config(selected,"");
    const auto launch=[&] {BLAS_CHECK(raw_launch(&choices[selected].algo));};
    launch(); CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    record_problem(out,p,a,b,c,std::abs(scalar.beta)!=0);
    validate_gemm(ctx,out,p,a,b,c,bias_type,scalar,dd.get(),epi,block_size,scale_a,scale_b);
    if(ctx.opts.get("compute")=="fixedpoint")record_fixedpoint_indicator(ctx,out,emulation_bits,"validation");
    if(out.validation!="passed") return;
    out.samples_ms=ctx.measure(launch);
    if(ctx.opts.get("compute")=="fixedpoint")record_fixedpoint_indicator(ctx,out,emulation_bits,"timing");
    validate_gemm(ctx,out,p,a,b,c,bias_type,scalar,dd.get(),epi,block_size,scale_a,scale_b);
    out.metadata["validation_timing_phase"]="candidate checks plus before and after independent performance measurement";
}

// The typed Level 1/2/3 interfaces deliberately use no GemmEx shortcuts.
template<class T> struct blas_api;
#define REAL_API(T, P, L) \
template<> struct blas_api<T> { \
    using real = T; \
    static cublasStatus_t axpy(cublasHandle_t h,int n,const T*a,const T*x,int ix,T*y,int iy){return cublas##P##axpy(h,n,a,x,ix,y,iy);} \
    static cublasStatus_t scal(cublasHandle_t h,int n,const T*a,T*x,int ix){return cublas##P##scal(h,n,a,x,ix);} \
    static cublasStatus_t dot(cublasHandle_t h,int n,const T*x,int ix,const T*y,int iy,T*z,bool){return cublas##P##dot(h,n,x,ix,y,iy,z);} \
    static cublasStatus_t nrm2(cublasHandle_t h,int n,const T*x,int ix,T*z){return cublas##P##nrm2(h,n,x,ix,z);} \
    static cublasStatus_t asum(cublasHandle_t h,int n,const T*x,int ix,T*z){return cublas##P##asum(h,n,x,ix,z);} \
    static cublasStatus_t iamax(cublasHandle_t h,int n,const T*x,int ix,int*z){return cublasI##L##amax(h,n,x,ix,z);} \
    static cublasStatus_t iamin(cublasHandle_t h,int n,const T*x,int ix,int*z){return cublasI##L##amin(h,n,x,ix,z);} \
    static cublasStatus_t copy(cublasHandle_t h,int n,const T*x,int ix,T*y,int iy){return cublas##P##copy(h,n,x,ix,y,iy);} \
    static cublasStatus_t swap(cublasHandle_t h,int n,T*x,int ix,T*y,int iy){return cublas##P##swap(h,n,x,ix,y,iy);} \
    static cublasStatus_t gemv(cublasHandle_t h,cublasOperation_t tr,int m,int n,const T*a,const T*A,int ld,const T*x,int ix,const T*b,T*y,int iy){return cublas##P##gemv(h,tr,m,n,a,A,ld,x,ix,b,y,iy);} \
    static cublasStatus_t geam(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,const T*a,const T*A,int la,const T*b,const T*B,int lb,T*C,int lc){return cublas##P##geam(h,ta,tb,m,n,a,A,la,b,B,lb,C,lc);} \
    static cublasStatus_t trsm(cublasHandle_t h,cublasSideMode_t side,cublasFillMode_t u,cublasOperation_t tr,cublasDiagType_t diag,int m,int n,const T*a,const T*A,int la,T*B,int lb){return cublas##P##trsm(h,side,u,tr,diag,m,n,a,A,la,B,lb);} \
    static cublasStatus_t trmm(cublasHandle_t h,cublasSideMode_t side,cublasFillMode_t u,cublasOperation_t tr,cublasDiagType_t diag,int m,int n,const T*a,const T*A,int la,const T*B,int lb,T*C,int lc){return cublas##P##trmm(h,side,u,tr,diag,m,n,a,A,la,B,lb,C,lc);} \
    static cublasStatus_t symm(cublasHandle_t h,cublasSideMode_t side,cublasFillMode_t u,int m,int n,const T*a,const T*A,int la,const T*B,int lb,const T*b,T*C,int lc){return cublas##P##symm(h,side,u,m,n,a,A,la,B,lb,b,C,lc);} \
    static cublasStatus_t hemm(cublasHandle_t,cublasSideMode_t,cublasFillMode_t,int,int,const T*,const T*,int,const T*,int,const T*,T*,int){return CUBLAS_STATUS_NOT_SUPPORTED;} \
    static cublasStatus_t syrk(cublasHandle_t h,cublasFillMode_t u,cublasOperation_t tr,int n,int k,const T*a,const T*A,int la,const T*b,T*C,int lc){return cublas##P##syrk(h,u,tr,n,k,a,A,la,b,C,lc);} \
};
REAL_API(float,S,s)
REAL_API(double,D,d)
#undef REAL_API
#define COMPLEX_API(T, R, P, PR, L) \
template<> struct blas_api<T> { \
    using real = R; \
    static cublasStatus_t axpy(cublasHandle_t h,int n,const T*a,const T*x,int ix,T*y,int iy){return cublas##P##axpy(h,n,a,x,ix,y,iy);} \
    static cublasStatus_t scal(cublasHandle_t h,int n,const T*a,T*x,int ix){return cublas##P##scal(h,n,a,x,ix);} \
    static cublasStatus_t dot(cublasHandle_t h,int n,const T*x,int ix,const T*y,int iy,T*z,bool conj){return conj?cublas##P##dotc(h,n,x,ix,y,iy,z):cublas##P##dotu(h,n,x,ix,y,iy,z);} \
    static cublasStatus_t nrm2(cublasHandle_t h,int n,const T*x,int ix,R*z){return cublas##PR##nrm2(h,n,x,ix,z);} \
    static cublasStatus_t asum(cublasHandle_t h,int n,const T*x,int ix,R*z){return cublas##PR##asum(h,n,x,ix,z);} \
    static cublasStatus_t iamax(cublasHandle_t h,int n,const T*x,int ix,int*z){return cublasI##L##amax(h,n,x,ix,z);} \
    static cublasStatus_t iamin(cublasHandle_t h,int n,const T*x,int ix,int*z){return cublasI##L##amin(h,n,x,ix,z);} \
    static cublasStatus_t copy(cublasHandle_t h,int n,const T*x,int ix,T*y,int iy){return cublas##P##copy(h,n,x,ix,y,iy);} \
    static cublasStatus_t swap(cublasHandle_t h,int n,T*x,int ix,T*y,int iy){return cublas##P##swap(h,n,x,ix,y,iy);} \
    static cublasStatus_t gemv(cublasHandle_t h,cublasOperation_t tr,int m,int n,const T*a,const T*A,int ld,const T*x,int ix,const T*b,T*y,int iy){return cublas##P##gemv(h,tr,m,n,a,A,ld,x,ix,b,y,iy);} \
    static cublasStatus_t geam(cublasHandle_t h,cublasOperation_t ta,cublasOperation_t tb,int m,int n,const T*a,const T*A,int la,const T*b,const T*B,int lb,T*C,int lc){return cublas##P##geam(h,ta,tb,m,n,a,A,la,b,B,lb,C,lc);} \
    static cublasStatus_t trsm(cublasHandle_t h,cublasSideMode_t side,cublasFillMode_t u,cublasOperation_t tr,cublasDiagType_t diag,int m,int n,const T*a,const T*A,int la,T*B,int lb){return cublas##P##trsm(h,side,u,tr,diag,m,n,a,A,la,B,lb);} \
    static cublasStatus_t trmm(cublasHandle_t h,cublasSideMode_t side,cublasFillMode_t u,cublasOperation_t tr,cublasDiagType_t diag,int m,int n,const T*a,const T*A,int la,const T*B,int lb,T*C,int lc){return cublas##P##trmm(h,side,u,tr,diag,m,n,a,A,la,B,lb,C,lc);} \
    static cublasStatus_t symm(cublasHandle_t h,cublasSideMode_t side,cublasFillMode_t u,int m,int n,const T*a,const T*A,int la,const T*B,int lb,const T*b,T*C,int lc){return cublas##P##symm(h,side,u,m,n,a,A,la,B,lb,b,C,lc);} \
    static cublasStatus_t hemm(cublasHandle_t h,cublasSideMode_t side,cublasFillMode_t u,int m,int n,const T*a,const T*A,int la,const T*B,int lb,const T*b,T*C,int lc){return cublas##P##hemm(h,side,u,m,n,a,A,la,B,lb,b,C,lc);} \
    static cublasStatus_t syrk(cublasHandle_t h,cublasFillMode_t u,cublasOperation_t tr,int n,int k,const T*a,const T*A,int la,const T*b,T*C,int lc){return cublas##P##syrk(h,u,tr,n,k,a,A,la,b,C,lc);} \
};
COMPLEX_API(cuComplex,float,C,Sc,c)
COMPLEX_API(cuDoubleComplex,double,Z,Dz,z)
#undef COMPLEX_API

template<class T> __host__ __device__ inline T real_scalar(double x,double y=0) { (void)y; return static_cast<T>(x); }
template<> __host__ __device__ inline cuComplex real_scalar<cuComplex>(double x,double y) {return make_cuComplex(x,y);}
template<> __host__ __device__ inline cuDoubleComplex real_scalar<cuDoubleComplex>(double x,double y) {return make_cuDoubleComplex(x,y);}

#include "blas_software.inc"

inline bool is_software_backend(const context& ctx) {
    return ctx.opts.get("backend")=="software"||ctx.opts.get("backend")=="emulated";
}

template<class T> void blas_vector(context& ctx,result& out,type_info dtype) {
    using api=blas_api<T>;
    const bool software=is_software_backend(ctx);
    out.execution=software?"software_blas_simt":"library_selected";
    out.metadata["execution_evidence"]=software?"custom CUDA SIMT kernel; no cuBLAS operation in timed callback":"API acceptance; instruction path requires SASS";
    const std::string op=ctx.opts.get("op","axpy");
    if(ctx.opts.has("m")||ctx.opts.has("k"))
        out.metadata["inactive_dimension_options"]=ctx.opts.has("m")?(ctx.opts.has("k")?"m,k":"m"):"k";
    const int n=positive_int(ctx.opts,ctx.opts.has("elements")?"elements":"n",1<<20);
    if(ctx.opts.has("elements")&&ctx.opts.has("n")&&ctx.opts.integer("n",n)!=n)
        throw std::invalid_argument("BLAS vector elements and n must agree when both are specified");
    const int ix=positive_int(ctx.opts,"incx",1),iy=positive_int(ctx.opts,"incy",1);
    const size_t nx=product(n,ix),ny=product(n,iy);
    ctx.memory_guard(product(sum_sizes({product(2,nx),product(2,ny),std::max(nx,ny),1}),sizeof(T)));
    device_buffer<T> x(nx),y(ny),saved(std::max(nx,ny)),saved_x(op=="swap"?nx:0),scalar(1);
    const T alpha=real_scalar<T>(ctx.opts.real("alpha",.75),ctx.opts.real("alpha-imag",0));
    const bool conjugate=ctx.opts.flag("conjugate",false);
    fill_buffer(ctx,x.get(),dtype,nx,17);fill_buffer(ctx,y.get(),dtype,ny,29);
    CUDA_CHECK(cudaMemsetAsync(scalar.get(),0,sizeof(T),ctx.stream));
    const bool inplace=op=="axpy"||op=="scal"||op=="swap";
    const bool reduction=op=="dot"||op=="nrm2"||op=="asum"||op=="iamax"||op=="iamin";
    if(inplace) CUDA_CHECK(cudaMemcpyAsync(saved.get(),op=="scal"?x.get():y.get(),(op=="scal"?nx:ny)*sizeof(T),cudaMemcpyDeviceToDevice,ctx.stream));
    if(op=="swap")CUDA_CHECK(cudaMemcpyAsync(saved_x.get(),x.get(),nx*sizeof(T),cudaMemcpyDeviceToDevice,ctx.stream));
    handle h(ctx.stream);
    if(reduction) BLAS_CHECK(cublasSetPointerMode(h.value,CUBLAS_POINTER_MODE_DEVICE));
    const auto restore=[&]{
        if(inplace)CUDA_CHECK(cudaMemcpyAsync(op=="scal"?x.get():y.get(),saved.get(),(op=="scal"?nx:ny)*sizeof(T),cudaMemcpyDeviceToDevice,ctx.stream));
        if(op=="swap")CUDA_CHECK(cudaMemcpyAsync(x.get(),saved_x.get(),nx*sizeof(T),cudaMemcpyDeviceToDevice,ctx.stream));
    };
    const auto launch=[&] {
        if(software) {sb_vector_launch(ctx,op,n,alpha,x.get(),ix,y.get(),iy,scalar.get(),conjugate);return;}
        if(op=="axpy") BLAS_CHECK(api::axpy(h.value,n,&alpha,x.get(),ix,y.get(),iy));
        else if(op=="scal") BLAS_CHECK(api::scal(h.value,n,&alpha,x.get(),ix));
        else if(op=="dot") BLAS_CHECK(api::dot(h.value,n,x.get(),ix,y.get(),iy,scalar.get(),conjugate));
        else if(op=="nrm2") BLAS_CHECK(api::nrm2(h.value,n,x.get(),ix,reinterpret_cast<typename api::real*>(scalar.get())));
        else if(op=="asum") BLAS_CHECK(api::asum(h.value,n,x.get(),ix,reinterpret_cast<typename api::real*>(scalar.get())));
        else if(op=="iamax") BLAS_CHECK(api::iamax(h.value,n,x.get(),ix,reinterpret_cast<int*>(scalar.get())));
        else if(op=="iamin") BLAS_CHECK(api::iamin(h.value,n,x.get(),ix,reinterpret_cast<int*>(scalar.get())));
        else if(op=="copy") BLAS_CHECK(api::copy(h.value,n,x.get(),ix,y.get(),iy));
        else if(op=="swap") BLAS_CHECK(api::swap(h.value,n,x.get(),ix,y.get(),iy));
        else throw std::invalid_argument("unknown BLAS vector operation");
    };
    launch();CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    bool good=true;
    auto tol=tolerance_for(dtype.id,"default",reduction?n:1);
    if(reduction) {
        T actual{}; CUDA_CHECK(cudaMemcpy(&actual,scalar.get(),sizeof(T),cudaMemcpyDeviceToHost));
        wide expected{};long double sum_abs=0;
        int expected_index=1;
        long double extremum=op=="iamin"?std::numeric_limits<long double>::infinity():-1;
        for(int i=0;i<n;++i) {
            auto xv=as_wide(make_value<T>(size_t(i)*ix,17));
            auto yv=as_wide(make_value<T>(size_t(i)*iy,29));
            if(op=="nrm2") {expected+=std::norm(xv);sum_abs+=std::norm(xv);}
            else if(op=="asum") {const auto norm=std::abs(xv.real())+std::abs(xv.imag());expected+=norm;sum_abs+=norm;}
            else if(op=="iamax"||op=="iamin") {
                const auto norm=std::abs(xv.real())+std::abs(xv.imag());
                if((op=="iamax"&&norm>extremum)||(op=="iamin"&&norm<extremum)){extremum=norm;expected_index=i+1;}
            } else {if(conjugate)xv=std::conj(xv);expected+=xv*yv;sum_abs+=std::abs(xv)*std::abs(yv);}
        }
        wide observed{};
        if(op=="iamax"||op=="iamin") {
            int index=0;std::memcpy(&index,&actual,sizeof(index));
            observed=index;expected=expected_index;tol={0,0,0,0};
        } else if(op=="nrm2"||op=="asum") {
            typename api::real norm{};std::memcpy(&norm,&actual,sizeof(norm));
            observed=norm;
            if(op=="nrm2")expected=std::sqrt(expected.real());
            sum_abs=expected.real();
        } else observed=as_wide(actual);
        good=compare_value(out,observed,expected,sum_abs,tol);
        out.metrics["validation_elements"]=n;
        out.metadata["validation_scope"]="full_CPU_reduction";
    } else {
        const auto indices=sample_indices(n,positive_int(ctx.opts,"validation-samples",128));
        const auto actual=gather_matrix(ctx,op=="scal"?static_cast<void*>(x.get()):static_cast<void*>(y.get()),dtype,
                                       indices,1,n,op=="scal"?ix:iy,op=="scal"?nx:ny);
        for(size_t j=0;j<indices.size();++j) {
            const auto i=indices[j];
            const auto xv=as_wide(make_value<T>(i*ix,17));
            const auto yv=as_wide(make_value<T>(i*iy,29));
            const auto expected=op=="copy"||op=="swap"?xv:as_wide(alpha)*xv+(op=="axpy"?yv:wide{});
            good=compare_value(out,{actual[j].x,actual[j].y},expected,std::abs(expected),tol)&&good;
        }
        if(op=="swap") {
            const auto actual_x=gather_matrix(ctx,x.get(),dtype,indices,1,n,ix,nx);
            for(size_t j=0;j<indices.size();++j) {
                const auto expected=as_wide(make_value<T>(indices[j]*iy,29));
                good=compare_value(out,{actual_x[j].x,actual_x[j].y},expected,std::abs(expected),tol)&&good;
            }
        }
        out.metrics["validation_elements"]=indices.size();
        out.metadata["validation_scope"]=indices.size()==size_t(n)?"all_output_elements":"deterministic_output_samples";
    }
    out.validation=good?"passed":"failed";
    if(!good){out.status="validation_failed";out.reason="BLAS vector reference mismatch";return;}
    out.samples_ms=inplace?measure_restored(ctx,restore,launch):ctx.measure(launch);
    out.work_per_iteration=(op=="copy"||op=="swap"?0:op=="axpy"?(dtype.complex?8:2):op=="scal"?(dtype.complex?6:1):
                            op=="dot"?(dtype.complex?8:2):op=="iamax"||op=="iamin"?(dtype.complex?4:2):(dtype.complex?4:2))*double(n);
    out.bytes_per_iteration=double(n)*sizeof(T)*(op=="axpy"?3:op=="swap"?4:reduction&&op!="dot"?1:2);
    out.execution=software?"software_blas_simt":"library_selected";out.unit=op=="copy"||op=="swap"?"GB/s":"Gop/s";
    if(software)out.metadata["implementation_policy"]="SIMT vector operations; reductions use one 256-thread block and a fixed reduction tree";
    out.metrics["effective_elements"]=n;
    out.metrics["incx"]=ix;
    out.metrics["incy"]=iy;
    out.metadata["timing"]=inplace?"individual_cuda_events_excluding_input_reset":"cuda_event_batch";
    int version=0;BLAS_CHECK(cublasGetVersion(h.value,&version));out.metrics["cublas_version"]=version;
}

template<class T> __host__ __device__ T triangular_value(int row,int col,int order,bool unit) {
    if(row==col) return real_scalar<T>(unit?1.0:2.0);
    const auto original=make_value<T>(size_t(col)*order+row,113);
    const auto v=as_double2(original);
    return real_scalar<T>(v.x/order,v.y/order);
}
template<class T> __global__ void triangular_fill(T* data,int order,int ld,bool unit) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(ld)*order;i+=size_t(gridDim.x)*blockDim.x) {
        const int row=i%ld,col=i/ld;
        data[i]=row<order?triangular_value<T>(row,col,order,unit):real_scalar<T>(0);
    }
}

template<class T> void blas_matrix(context& ctx,result& out,type_info dtype) {
    using api=blas_api<T>;
    const bool software=is_software_backend(ctx);
    out.execution=software?"software_blas_simt":"library_selected";
    out.metadata["execution_evidence"]=software?"custom CUDA SIMT kernel; no cuBLAS operation in timed callback":"API acceptance; instruction path requires SASS";
    const std::string op=ctx.opts.get("op","gemv");
    if(op=="syrk"&&ctx.opts.has("m"))out.metadata["inactive_dimension_options"]="m; SYRK dimensions are n and k";
    if(op!="syrk"&&ctx.opts.has("k"))out.metadata["inactive_dimension_options"]="k; this BLAS operation uses m and n";
    const int m=positive_int(ctx.opts,"m",1024),n=positive_int(ctx.opts,"n",1024),k=positive_int(ctx.opts,"k",1024);
    const auto pad64=ctx.opts.integer("ld-pad",0);
    if(pad64<0||pad64>INT32_MAX-std::max({m,n,k}))throw std::invalid_argument("invalid ld-pad");
    const int pad=static_cast<int>(pad64);
    const auto ta=transpose(ctx.opts.get("trans-a","N")),tb=transpose(ctx.opts.get("trans-b","N"));
    const auto uplo_text=ctx.opts.get("uplo","lower");
    if(uplo_text!="lower"&&uplo_text!="upper")throw std::invalid_argument("uplo must be lower or upper");
    const auto uplo=uplo_text=="lower"?CUBLAS_FILL_MODE_LOWER:CUBLAS_FILL_MODE_UPPER;
    const auto side_text=ctx.opts.get("side","left");
    if(side_text!="left"&&side_text!="right")throw std::invalid_argument("side must be left or right");
    const auto side=side_text=="left"?CUBLAS_SIDE_LEFT:CUBLAS_SIDE_RIGHT;
    const auto diag_text=ctx.opts.get("diag","nonunit");
    if(diag_text!="unit"&&diag_text!="nonunit")throw std::invalid_argument("diag must be unit or nonunit");
    const bool unit=diag_text=="unit";
    const auto diag=unit?CUBLAS_DIAG_UNIT:CUBLAS_DIAG_NON_UNIT;
    const T alpha=real_scalar<T>(ctx.opts.real("alpha",.75),ctx.opts.real("alpha-imag",0));
    const T beta=real_scalar<T>(ctx.opts.real("beta",.25),ctx.opts.real("beta-imag",0));
    const bool square_a=op=="trsm"||op=="trmm"||op=="symm"||op=="hemm";
    if((op=="symm"||op=="hemm")&&ta!=CUBLAS_OP_N)
        throw std::invalid_argument("symm/hemm do not accept trans-a; use N");
    const int order=square_a?(side==CUBLAS_SIDE_LEFT?m:n):n;
    const int lda=op=="gemv"?m+pad:op=="geam"?(ta==CUBLAS_OP_N?m:n)+pad:
                  square_a?order+pad:(ta==CUBLAS_OP_N?n:k)+pad;
    const int ldb=op=="geam"?(tb==CUBLAS_OP_N?m:n)+pad:m+pad;
    const int ldc=op=="syrk"?n+pad:m+pad;
    const size_t na=op=="gemv"?product(lda,n):op=="geam"?product(lda,ta==CUBLAS_OP_N?n:m):
                    square_a?product(lda,order):product(lda,ta==CUBLAS_OP_N?k:n);
    const size_t nb=op=="geam"?product(ldb,tb==CUBLAS_OP_N?n:m):product(ldb,n);
    const int ix=positive_int(ctx.opts,"incx",1),iy=positive_int(ctx.opts,"incy",1);
    const int xlength=ta==CUBLAS_OP_N?n:m,ylength=ta==CUBLAS_OP_N?m:n;
    const size_t nx=product(xlength,ix),ny=product(ylength,iy);
    const size_t nc=op=="gemv"?ny:op=="syrk"?product(ldc,n):product(ldc,n);
    ctx.memory_guard(product(sum_sizes({na,nb,product(2,nc),nx}),sizeof(T)));
    device_buffer<T> a(na),b(nb),c(nc),saved(nc),x(nx);
    fill_buffer(ctx,a.get(),dtype,na,17);fill_buffer(ctx,b.get(),dtype,nb,29);
    fill_buffer(ctx,c.get(),dtype,nc,71);fill_buffer(ctx,x.get(),dtype,nx,53);
    if(square_a) {
        triangular_fill<T><<<std::min<size_t>((na+255)/256,4096),256,0,ctx.stream>>>(a.get(),order,lda,unit);
        CUDA_CHECK(cudaGetLastError());
    }
    if(op=="trsm") {
        CUDA_CHECK(cudaMemcpyAsync(c.get(),b.get(),nc*sizeof(T),cudaMemcpyDeviceToDevice,ctx.stream));
    }
    const bool inplace=op!="geam"&&op!="trmm";
    if(inplace)CUDA_CHECK(cudaMemcpyAsync(saved.get(),c.get(),nc*sizeof(T),cudaMemcpyDeviceToDevice,ctx.stream));
    handle h(ctx.stream);
    const auto restore=[&]{if(inplace)CUDA_CHECK(cudaMemcpyAsync(c.get(),saved.get(),nc*sizeof(T),cudaMemcpyDeviceToDevice,ctx.stream));};
    const auto launch=[&]{
        if(software) {sb_matrix_launch(ctx,op,ta,tb,side,uplo,diag,m,n,k,alpha,a.get(),lda,b.get(),ldb,beta,c.get(),ldc,x.get(),ix,iy);return;}
        if(op=="gemv")BLAS_CHECK(api::gemv(h.value,ta,m,n,&alpha,a.get(),lda,x.get(),ix,&beta,c.get(),iy));
        else if(op=="geam")BLAS_CHECK(api::geam(h.value,ta,tb,m,n,&alpha,a.get(),lda,&beta,b.get(),ldb,c.get(),ldc));
        else if(op=="trsm")BLAS_CHECK(api::trsm(h.value,side,uplo,ta,diag,m,n,&alpha,a.get(),lda,c.get(),ldc));
        else if(op=="trmm")BLAS_CHECK(api::trmm(h.value,side,uplo,ta,diag,m,n,&alpha,a.get(),lda,b.get(),ldb,c.get(),ldc));
        else if(op=="symm")BLAS_CHECK(api::symm(h.value,side,uplo,m,n,&alpha,a.get(),lda,b.get(),ldb,&beta,c.get(),ldc));
        else if(op=="hemm")BLAS_CHECK(api::hemm(h.value,side,uplo,m,n,&alpha,a.get(),lda,b.get(),ldb,&beta,c.get(),ldc));
        else if(op=="syrk")BLAS_CHECK(api::syrk(h.value,uplo,ta,n,k,&alpha,a.get(),lda,&beta,c.get(),ldc));
        else throw std::invalid_argument("unknown BLAS matrix operation");
    };
    launch();CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    const int rows=op=="gemv"?1:op=="syrk"?n:m,cols=op=="gemv"?ylength:n;
    auto indices=sample_indices(product(rows,cols),positive_int(ctx.opts,"validation-samples",128));
    if(op=="syrk") {
        for(auto& t:indices) {auto row=t%n,col=t/n;if((uplo==CUBLAS_FILL_MODE_LOWER&&row<col)||(uplo==CUBLAS_FILL_MODE_UPPER&&row>col))t=row*size_t(n)+col;}
    }
    const auto actual=gather_matrix(ctx,c.get(),dtype,indices,rows,cols,op=="gemv"?iy:ldc,nc);
    std::vector<T> solution;
    if(op=="trsm") {solution.resize(nc);CUDA_CHECK(cudaMemcpy(solution.data(),c.get(),nc*sizeof(T),cudaMemcpyDeviceToHost));}
    bool good=true;
    const auto tol=tolerance_for(dtype.id,"default",square_a?order:op=="syrk"?k:std::max(m,n));
    for(size_t s=0;s<indices.size();++s) {
        const int row=indices[s]%rows,col=indices[s]/rows;
        wide expected{},observed(actual[s].x,actual[s].y);long double sum_abs=0;
        if(op=="geam") {
            auto av=as_wide(make_value<T>(ta==CUBLAS_OP_N?size_t(col)*lda+row:size_t(row)*lda+col,17));
            auto bv=as_wide(make_value<T>(tb==CUBLAS_OP_N?size_t(col)*ldb+row:size_t(row)*ldb+col,29));
            if(ta==CUBLAS_OP_C)av=std::conj(av);
            if(tb==CUBLAS_OP_C)bv=std::conj(bv);
            expected=as_wide(alpha)*av+as_wide(beta)*bv;sum_abs=std::abs(as_wide(alpha)*av)+std::abs(as_wide(beta)*bv);
        } else if(op=="gemv") {
            wide sum{};
            for(int r=0;r<xlength;++r) {
                auto av=as_wide(make_value<T>(ta==CUBLAS_OP_N?size_t(r)*lda+col:size_t(col)*lda+r,17));
                if(ta==CUBLAS_OP_C)av=std::conj(av);
                const auto xv=as_wide(make_value<T>(size_t(r)*ix,53));sum+=av*xv;sum_abs+=std::abs(av)*std::abs(xv);
            }
            const auto cv=as_wide(make_value<T>(size_t(col)*iy,71));
            expected=as_wide(alpha)*sum+as_wide(beta)*cv;
            sum_abs=std::abs(as_wide(alpha))*sum_abs+std::abs(as_wide(beta)*cv);
        } else if(op=="syrk") {
            wide sum{};
            for(int r=0;r<k;++r) {
                auto av=as_wide(make_value<T>(ta==CUBLAS_OP_N?size_t(r)*lda+row:size_t(row)*lda+r,17));
                auto bv=as_wide(make_value<T>(ta==CUBLAS_OP_N?size_t(r)*lda+col:size_t(col)*lda+r,17));
                sum+=av*bv;sum_abs+=std::abs(av)*std::abs(bv);
            }
            const auto cv=as_wide(make_value<T>(size_t(col)*ldc+row,71));
            expected=as_wide(alpha)*sum+as_wide(beta)*cv;
            sum_abs=std::abs(as_wide(alpha))*sum_abs+std::abs(as_wide(beta)*cv);
        } else if(op=="trmm"||op=="symm"||op=="hemm") {
            wide sum{};
            for(int r=0;r<order;++r) {
                int ar=side==CUBLAS_SIDE_LEFT?row:r,ac=side==CUBLAS_SIDE_LEFT?r:col;
                bool conjugate=false;
                if(op=="trmm") {
                    if(ta!=CUBLAS_OP_N)std::swap(ar,ac);
                    if((uplo==CUBLAS_FILL_MODE_LOWER&&ar<ac)||(uplo==CUBLAS_FILL_MODE_UPPER&&ar>ac))continue;
                    conjugate=ta==CUBLAS_OP_C;
                } else if((uplo==CUBLAS_FILL_MODE_LOWER&&ar<ac)||(uplo==CUBLAS_FILL_MODE_UPPER&&ar>ac)) {
                    std::swap(ar,ac);conjugate=op=="hemm";
                }
                auto av=as_wide(triangular_value<T>(ar,ac,order,unit));
                if(conjugate)av=std::conj(av);
                const auto bv=as_wide(make_value<T>(side==CUBLAS_SIDE_LEFT?size_t(col)*ldb+r:size_t(r)*ldb+row,29));
                sum+=av*bv;sum_abs+=std::abs(av)*std::abs(bv);
            }
            expected=as_wide(alpha)*sum;sum_abs*=std::abs(as_wide(alpha));
            if(op!="trmm") {
                const auto cv=as_wide(beta)*as_wide(make_value<T>(size_t(col)*ldc+row,71));
                expected+=cv;sum_abs+=std::abs(cv);
            }
        } else {
            // Verify triangular solve by the residual op(A)X-alpha B (or Xop(A)).
            wide lhs{};
            for(int r=0;r<order;++r) {
                int ar=side==CUBLAS_SIDE_LEFT?row:r,ac=side==CUBLAS_SIDE_LEFT?r:col;
                if(ta!=CUBLAS_OP_N)std::swap(ar,ac);
                if((uplo==CUBLAS_FILL_MODE_LOWER&&ar<ac)||(uplo==CUBLAS_FILL_MODE_UPPER&&ar>ac))continue;
                auto av=as_wide(triangular_value<T>(ar,ac,order,unit));
                if(ta==CUBLAS_OP_C)av=std::conj(av);
                const auto xv=as_wide(solution[side==CUBLAS_SIDE_LEFT?size_t(col)*ldc+r:size_t(r)*ldc+row]);
                lhs+=av*xv;sum_abs+=std::abs(av)*std::abs(xv);
            }
            expected=as_wide(alpha)*as_wide(make_value<T>(size_t(col)*ldb+row,29));observed=lhs;
        }
        good=compare_value(out,observed,expected,sum_abs,tol)&&good;
    }
    out.validation=good?"passed":"failed";
    if(!good){out.status="validation_failed";out.reason="BLAS matrix reference mismatch";return;}
    out.samples_ms=inplace?measure_restored(ctx,restore,launch):ctx.measure(launch);
    const double mac=dtype.complex?8.0:2.0;
    out.work_per_iteration=op=="gemv"?mac*m*n:op=="geam"?(dtype.complex?14.0:3.0)*m*n:
                           op=="syrk"?mac*n*(double(n)+1)*.5*k:
                           op=="symm"||op=="hemm"?mac*m*n*order:(dtype.complex?4.0:1.0)*m*n*order;
    out.bytes_per_iteration=(op=="gemv"?double(m)*n+xlength+2.0*ylength:
                             op=="geam"?3.0*m*n:
                             op=="syrk"?double(n)*k+double(n)*(double(n)+1):
                             double(order)*(double(order)+1)/2+(op=="symm"||op=="hemm"?3.0:2.0)*m*n)*sizeof(T);
    out.metadata["byte_count"]="useful tensor traffic estimate; not measured DRAM bytes";
    out.execution=software?"software_blas_simt":"library_selected";out.unit="Gop/s";
    if(software)out.metadata["implementation_policy"]="SIMT matrix outputs; triangular solve uses one sequential substitution per independent right-hand side";
    out.metadata["timing"]=inplace?"individual_cuda_events_excluding_input_reset":"cuda_event_batch";
    out.metadata["reference"]=op=="trsm"?"CPU long-double triangular equation residual":"CPU long-double reference";
    out.metadata["validation_scope"]="deterministic_output_samples";
    out.metrics["validation_elements"]=indices.size();
    int version=0;BLAS_CHECK(cublasGetVersion(h.value,&version));out.metrics["cublas_version"]=version;
}

template<class T> void run_typed_blas(context& ctx,result& out,type_info dtype) {
    const auto op=ctx.opts.get("op","axpy");
    if(op=="axpy"||op=="scal"||op=="dot"||op=="nrm2"||op=="asum"||op=="iamax"||op=="iamin"||op=="copy"||op=="swap")blas_vector<T>(ctx,out,dtype);
    else if(op=="gemv"||op=="geam"||op=="trsm"||op=="syrk"||op=="trmm"||op=="symm"||op=="hemm")blas_matrix<T>(ctx,out,dtype);
    else skip(out,"unknown BLAS operation in this backend: "+op,"not_implemented");
}

} // namespace gemm_detail

bool run_gemm(context& ctx,result& out) {
    const auto family=ctx.opts.get("family");
    if(family!="gemm"&&family!="blas")return false;
    try {
        if(family=="gemm") {
            const auto backend=ctx.opts.get("backend","cublas");
            if(backend=="software"||backend=="emulated") {
                gemm_detail::software_gemm(ctx,out);return true;
            }
            if(backend!="cublas"&&backend!="cublaslt") {
                skip(out,"GEMM backend must be cublas, cublaslt, or software (alias emulated)","not_implemented");
                return true;
            }
            if(ctx.opts.get("compute")=="bf16x9"&&
               !gemm_detail::cublas_bf16x9_supported(ctx.prop.major,ctx.prop.minor)) {
                const auto capability=std::to_string(ctx.prop.major)+"."+std::to_string(ctx.prop.minor);
                skip(out,"The BF16x9 policy API is available in CUDA 13.5, but supported BF16x9 execution "
                         "is limited to compute capabilities 10.0 and 10.3; selected device has compute "
                         "capability "+capability+". A library fallback is not BF16x9 coverage.");
                out.execution="not_executed";
                out.metadata["backend"]=backend;
                out.metadata["requested_compute"]="bf16x9";
                out.metadata["compute_policy_support"]="api_available_architecture_unsupported";
                out.metadata["capability_baseline"]="CUDA 13.5 cuBLAS";
                out.metadata["bf16x9_supported_compute_capabilities"]="10.0,10.3";
                out.metadata["support_evidence"]="documented_architecture_support_table";
                out.metadata["support_source"]="https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation";
                return true;
            }
            if(backend=="cublas")gemm_detail::cublas_gemm(ctx,out);
            else gemm_detail::lt_gemm(ctx,out);
        } else {
            const auto backend=ctx.opts.get("backend","cublas");
            if(backend!="cublas"&&backend!="native"&&backend!="software"&&backend!="emulated") {
                skip(out,"BLAS backend must be cublas (alias native) or software (alias emulated)","not_implemented");return true;
            }
            const auto dtype=gemm_detail::parse_type(ctx.opts.get("dtype","f32"));
            using namespace gemm_detail;
            if(dtype.id==kind::f32)run_typed_blas<float>(ctx,out,dtype);
            else if(dtype.id==kind::f64)run_typed_blas<double>(ctx,out,dtype);
            else if(dtype.id==kind::c64)run_typed_blas<cuComplex>(ctx,out,dtype);
            else if(dtype.id==kind::c128)run_typed_blas<cuDoubleComplex>(ctx,out,dtype);
            else skip(out,"typed BLAS suite supports f32, f64, c64, c128; use GEMM for mixed precision");
        }
    } catch(const gemm_detail::library_failure& error) {
        if(error.status==CUBLAS_STATUS_NOT_SUPPORTED||error.status==CUBLAS_STATUS_ARCH_MISMATCH||
           error.status==CUBLAS_STATUS_INVALID_VALUE) {
            skip(out,error.what());out.metadata["support_evidence"]="library_rejected_descriptor; not proof of hardware absence";
        } else throw;
    }
    return true;
}
} // namespace archbench
