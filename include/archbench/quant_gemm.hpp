// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cmath>
#if defined(__CUDACC__)
#define ARCHBENCH_QUANT_HD __host__ __device__
#else
#define ARCHBENCH_QUANT_HD
#endif
namespace archbench::quant_baseline {
// FP32 stored activations and decoded weights. Padding is excluded from K.
ARCHBENCH_QUANT_HD inline float decoded_dot(const float* a,const float* b,
        int row,int col,int k,int padded_k) {
    float sum=0;
    for(int p=0;p<k;++p)
        sum=fmaf(a[std::size_t(row)*k+p],b[std::size_t(col)*padded_k+p],sum);
    return sum;
}
}
#undef ARCHBENCH_QUANT_HD
