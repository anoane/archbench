// SPDX-License-Identifier: AGPL-3.0-or-later
// Compile-only probe for both binary forms used by src/alu.cu.
#include <cuda_runtime.h>
extern "C" __global__ void archbench_probe_binary(unsigned* out, unsigned a, unsigned b) {
    unsigned d[4] = {};
#define STEP(OP) asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32." OP ".popc " \
    "{%0,%1,%2,%3}, {%4,%4,%4,%4}, {%5,%5}, {%0,%1,%2,%3};" \
    : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a), "r"(b))
    STEP("and");
    STEP("xor");
#undef STEP
    for (int i = 0; i < 4; ++i) out[threadIdx.x * 4 + i] = d[i];
}
