// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once
#include <cstddef>
#if defined(__CUDACC__)
#define ARCHBENCH_BLAS_HD __host__ __device__
#else
#define ARCHBENCH_BLAS_HD
#endif

namespace archbench::software_blas {
enum class operation { axpy, scal, dot, nrm2, asum, iamax, iamin, copy, swap,
                       gemv, geam, trsm, trmm, symm, hemm, syrk };

// Ops supplies arithmetic on a real/complex stored scalar. The same index and
// solve algorithms are compiled by host tests and by the CUDA implementation.
// Transpose codes are 0=N, 1=T, 2=conjugate transpose, independent of cuBLAS enums.
template<class Ops>
ARCHBENCH_BLAS_HD typename Ops::value at(const typename Ops::value* data,
        int ld, int row, int col, int trans) {
    auto value = data[trans == 0 ? std::size_t(col) * ld + row : std::size_t(row) * ld + col];
    return trans == 2 ? Ops::conj(value) : value;
}

template<class Ops>
ARCHBENCH_BLAS_HD typename Ops::value triangular(const typename Ops::value* data,
        int ld, int row, int col, int trans, bool lower, bool unit) {
    if (row == col && unit) return Ops::one();
    const int stored_row = trans == 0 ? row : col;
    const int stored_col = trans == 0 ? col : row;
    if ((lower && stored_row < stored_col) || (!lower && stored_row > stored_col)) return Ops::zero();
    return at<Ops>(data, ld, row, col, trans);
}

template<class Ops>
ARCHBENCH_BLAS_HD typename Ops::value symmetric(const typename Ops::value* data,
        int ld, int row, int col, bool lower, bool hermitian) {
    const bool mirror = (lower && row < col) || (!lower && row > col);
    auto value = data[mirror ? std::size_t(row) * ld + col : std::size_t(col) * ld + row];
    if (hermitian && row == col) return Ops::real_part(value);
    return hermitian && mirror ? Ops::conj(value) : value;
}

template<class Ops>
ARCHBENCH_BLAS_HD typename Ops::value matrix_element(operation op, int row, int col,
        int m, int n, int k, int ta, int tb, bool left, bool lower, bool unit,
        typename Ops::value alpha, const typename Ops::value* a, int lda,
        const typename Ops::value* b, int ldb, typename Ops::value beta,
        const typename Ops::value* c, int ldc, const typename Ops::value* x, int ix, int iy) {
    using V = typename Ops::value;
    V sum = Ops::zero();
    if (op == operation::geam) {
        return Ops::add(Ops::mul(alpha, at<Ops>(a, lda, row, col, ta)),
                        Ops::mul(beta, at<Ops>(b, ldb, row, col, tb)));
    }
    if (op == operation::gemv) {
        const int length = ta == 0 ? n : m;
        for (int p = 0; p < length; ++p)
            sum = Ops::madd(at<Ops>(a, lda, row, p, ta), x[std::size_t(p) * ix], sum);
        return Ops::add(Ops::mul(alpha, sum), Ops::mul(beta, c[std::size_t(row) * iy]));
    }
    if (op == operation::syrk) {
        // Symmetric rank-k is A*A^T, not the Hermitian A*A^H operation.
        const int transpose_only = ta == 0 ? 0 : 1;
        for (int p = 0; p < k; ++p)
            sum = Ops::madd(at<Ops>(a, lda, row, p, transpose_only),
                            at<Ops>(a, lda, col, p, transpose_only), sum);
    } else {
        const int order = left ? m : n;
        for (int p = 0; p < order; ++p) {
            const int ar = left ? row : p, ac = left ? p : col;
            const V av = op == operation::trmm ? triangular<Ops>(a, lda, ar, ac, ta, lower, unit)
                : symmetric<Ops>(a, lda, ar, ac, lower, op == operation::hemm);
            const V bv = b[left ? std::size_t(col) * ldb + p : std::size_t(p) * ldb + row];
            sum = Ops::madd(av, bv, sum);
        }
    }
    const V scaled = Ops::mul(alpha, sum);
    return op == operation::trmm ? scaled : Ops::add(scaled, Ops::mul(beta, c[std::size_t(col) * ldc + row]));
}

template<class Ops>
ARCHBENCH_BLAS_HD void solve_rhs(int rhs, int m, int n, int trans, bool left,
        bool lower, bool unit, typename Ops::value alpha,
        const typename Ops::value* a, int lda, typename Ops::value* c, int ldc) {
    const int order = left ? m : n;
    const bool op_lower = trans == 0 ? lower : !lower;
    // For X*A=B, solve A^T*X^T=B^T. This changes traversal, not conjugation.
    const bool forward = left ? op_lower : !op_lower;
    for (int step = 0; step < order; ++step) {
        const int current = forward ? step : order - 1 - step;
        const auto output = left ? std::size_t(rhs) * ldc + current : std::size_t(current) * ldc + rhs;
        auto value = Ops::mul(alpha, c[output]);
        for (int previous = 0; previous < step; ++previous) {
            const int p = forward ? previous : order - 1 - previous;
            const auto coefficient = left ? triangular<Ops>(a, lda, current, p, trans, lower, unit)
                                          : triangular<Ops>(a, lda, p, current, trans, lower, unit);
            const auto solved = c[left ? std::size_t(rhs) * ldc + p : std::size_t(p) * ldc + rhs];
            value = Ops::sub(value, Ops::mul(coefficient, solved));
        }
        if (!unit) value = Ops::div(value, triangular<Ops>(a, lda, current, current, trans, lower, false));
        c[output] = value;
    }
}
} // namespace archbench::software_blas
#undef ARCHBENCH_BLAS_HD
