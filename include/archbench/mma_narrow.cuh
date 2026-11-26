// SPDX-License-Identifier: AGPL-3.0-or-later
// Included inside alu.cu's anonymous namespace after tensor_format is declared.
// Requires an architecture-specific sm_120a compilation, not merely sm_120.
// The expanded f8f6f4 path uses 8-bit containers: FP6 low 6 bits; FP4 bits 5:2.

template <tensor_format A, tensor_format B, bool Scaled>
struct narrow_mma;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 && defined(ARCHBENCH_ARCH_SUFFIX_A)
#define ARCHBENCH_NARROW_PAIR(A, ATYPE, B, BTYPE) \
template <bool Scaled> struct narrow_mma<tensor_format::A, tensor_format::B, Scaled> { \
    __device__ __forceinline__ static void step(float* d, uint32_t a, uint32_t b, uint32_t sa, uint32_t sb) { \
        if constexpr (!Scaled) { \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32." ATYPE "." BTYPE ".f32 " \
                "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};" \
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(b)); \
        } else { \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32." ATYPE "." BTYPE ".f32.ue8m0 " \
                "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3}, %6, {0,0}, %7, {0,0};" \
                : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(b), "r"(sa), "r"(sb)); \
        } \
    } \
    __device__ __forceinline__ static void step_half(uint32_t* d, uint32_t a, uint32_t b) { \
        asm volatile( \
            "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16." ATYPE "." BTYPE ".f16 " \
            "{%0,%1}, {%2,%2,%2,%2}, {%3,%3}, {%0,%1};" \
            : "+r"(d[0]), "+r"(d[1]) : "r"(a), "r"(b)); \
    } \
};
#else
#define ARCHBENCH_NARROW_PAIR(A, ATYPE, B, BTYPE) \
template <bool Scaled> struct narrow_mma<tensor_format::A, tensor_format::B, Scaled> { \
    __device__ __forceinline__ static void step(float*, uint32_t, uint32_t, uint32_t, uint32_t) {} \
    __device__ __forceinline__ static void step_half(uint32_t*, uint32_t, uint32_t) {} \
};
#endif

ARCHBENCH_NARROW_PAIR(e2m1, "e2m1", e2m1, "e2m1")
ARCHBENCH_NARROW_PAIR(e2m1, "e2m1", e2m3, "e2m3")
ARCHBENCH_NARROW_PAIR(e2m1, "e2m1", e3m2, "e3m2")
ARCHBENCH_NARROW_PAIR(e2m1, "e2m1", e4m3, "e4m3")
ARCHBENCH_NARROW_PAIR(e2m1, "e2m1", e5m2, "e5m2")
ARCHBENCH_NARROW_PAIR(e2m3, "e2m3", e2m1, "e2m1")
ARCHBENCH_NARROW_PAIR(e2m3, "e2m3", e2m3, "e2m3")
ARCHBENCH_NARROW_PAIR(e2m3, "e2m3", e3m2, "e3m2")
ARCHBENCH_NARROW_PAIR(e2m3, "e2m3", e4m3, "e4m3")
ARCHBENCH_NARROW_PAIR(e2m3, "e2m3", e5m2, "e5m2")
ARCHBENCH_NARROW_PAIR(e3m2, "e3m2", e2m1, "e2m1")
ARCHBENCH_NARROW_PAIR(e3m2, "e3m2", e2m3, "e2m3")
ARCHBENCH_NARROW_PAIR(e3m2, "e3m2", e3m2, "e3m2")
ARCHBENCH_NARROW_PAIR(e3m2, "e3m2", e4m3, "e4m3")
ARCHBENCH_NARROW_PAIR(e3m2, "e3m2", e5m2, "e5m2")
ARCHBENCH_NARROW_PAIR(e4m3, "e4m3", e2m1, "e2m1")
ARCHBENCH_NARROW_PAIR(e4m3, "e4m3", e2m3, "e2m3")
ARCHBENCH_NARROW_PAIR(e4m3, "e4m3", e3m2, "e3m2")
ARCHBENCH_NARROW_PAIR(e4m3, "e4m3", e4m3, "e4m3")
ARCHBENCH_NARROW_PAIR(e4m3, "e4m3", e5m2, "e5m2")
ARCHBENCH_NARROW_PAIR(e5m2, "e5m2", e2m1, "e2m1")
ARCHBENCH_NARROW_PAIR(e5m2, "e5m2", e2m3, "e2m3")
ARCHBENCH_NARROW_PAIR(e5m2, "e5m2", e3m2, "e3m2")
ARCHBENCH_NARROW_PAIR(e5m2, "e5m2", e4m3, "e4m3")
ARCHBENCH_NARROW_PAIR(e5m2, "e5m2", e5m2, "e5m2")

#undef ARCHBENCH_NARROW_PAIR

template <bool NvidiaScale>
__device__ __forceinline__ void packed_fp4_mma(float* d, uint32_t a, uint32_t b, uint32_t sa, uint32_t sb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 && defined(ARCHBENCH_ARCH_SUFFIX_A)
    if constexpr (!NvidiaScale) {
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3}, %6, {0,0}, %7, {0,0};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(b), "r"(sa), "r"(sb));
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3}, %6, {0,0}, %7, {0,0};"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a), "r"(b), "r"(sa), "r"(sb));
    }
#endif
}
