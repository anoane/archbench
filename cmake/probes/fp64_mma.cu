// SPDX-License-Identifier: AGPL-3.0-or-later
// Compile-only probe. Acceptance is not a runtime or native-SASS claim.
#include <cuda_runtime.h>
extern "C" __global__ void archbench_probe_fp64(double* out, double a, double b) {
    double d[2] = {};
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0,%1}, {%2}, {%3}, {%0,%1};"
        : "+d"(d[0]), "+d"(d[1]) : "d"(a), "d"(b));
    out[threadIdx.x * 2] = d[0];
    out[threadIdx.x * 2 + 1] = d[1];
}
