// SPDX-License-Identifier: AGPL-3.0-or-later
#include "archbench/quant_gemm.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    // Independent dense reference plus poisoned padding catches accidental use
    // of group-rounded K, row/column confusion, and loss of decoded weights.
    int checked=0;
    for(int k : {1,3,31,32,33,65,127}) {
        constexpr int m=5,n=7;
        const int padded=((k+31)/32)*32;
        std::vector<float> a(m*k), b(n*padded,std::numeric_limits<float>::quiet_NaN());
        for(int r=0;r<m;++r)for(int p=0;p<k;++p)a[r*k+p]=float((r*7+p*3)%29-14)/16;
        for(int c=0;c<n;++c)for(int p=0;p<k;++p)b[c*padded+p]=float((c*11+p*5)%31-15)/8;
        for(int r=0;r<m;++r)for(int c=0;c<n;++c) {
            long double expected=0;
            for(int p=0;p<k;++p)expected+=static_cast<long double>(a[r*k+p])*b[c*padded+p];
            const float got=archbench::quant_baseline::decoded_dot(a.data(),b.data(),r,c,k,padded);
            if(!std::isfinite(got)||std::abs(static_cast<long double>(got)-expected)>1e-5L)
                throw std::runtime_error("decoded GEMM differs from independent dense reference");
            ++checked;
        }
    }
    const float a[]={1,2},b[]={std::numeric_limits<float>::quiet_NaN(),1};
    if(!std::isnan(archbench::quant_baseline::decoded_dot(a,b,0,0,2,2)))
        throw std::runtime_error("nonfinite active input was hidden");
    std::cout<<"PASS "<<checked<<" decoded products; poisoned padding and active NaN\n";
}
