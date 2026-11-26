// SPDX-License-Identifier: AGPL-3.0-or-later
// Compile-only probe for every signedness pair used by src/alu.cu.
#include <cuda_runtime.h>
extern "C" __global__ void archbench_probe_int4(unsigned* out, unsigned a, unsigned b) {
    unsigned d[4] = {};
#define STEP(A, B) asm volatile("mma.sync.aligned.m16n8k64.row.col.s32." A "." B ".s32 " \
    "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};" \
    : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a), "r"(b))
    STEP("s4", "s4");
    STEP("s4", "u4");
    STEP("u4", "s4");
    STEP("u4", "u4");
#undef STEP
    for (int i = 0; i < 4; ++i) out[threadIdx.x * 4 + i] = d[i];
}
