// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

namespace archbench {
namespace gemm_detail {

// CUDA 13.5 cuBLAS, section 1.5, Floating Point Emulation:
// https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation
// Policy/API availability does not imply architecture support. Keep the exact
// documented set; a future capability must be reviewed before it is admitted.
constexpr bool cublas_bf16x9_supported(int major, int minor) noexcept {
    return major == 10 && (minor == 0 || minor == 3);
}

} // namespace gemm_detail
} // namespace archbench
