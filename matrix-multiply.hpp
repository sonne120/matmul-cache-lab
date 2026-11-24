#pragma once

//
// Function                        Storage         Outer tile order  Tile size  k per pass
// ------------------------------  --------------  ----------------  ---------  ----------
// bad_matrix_multiply             row pointers    none (i-j-k)      -          1
// fast_matrix_multiply            row pointers    none (i-k-j)      -          1
// matmul_tiled                    row pointers    ii-kk-jj          BS         1
// matmul_tiled_flat               flat row-major  ii-kk-jj          argument   1
// matmul_tiled_flat_v2            flat row-major  ii-jj-kk          argument   1
// blocked_matrix_multiply         flat row-major  si-sk-sj          BS         1
// blocked_k4_matrix_multiply      flat row-major  sj-si-sk          BS         4
// blocked_k4_ikj_matrix_multiply  flat row-major  ii-kk-jj          BS         4
// blocked_matrix_multiply_k8      flat row-major  si-sj-sk          argument   8
// ikj_k4_matrix_multiply          flat row-major  none (i-k-j)      -          4
// neon_4x4_matrix_multiply        flat row-major  jj-kk, packed B   BS         4x4 NEON kernel
// accelerate_dgemm_matrix_multiply  flat row-major  Apple Accelerate cblas_dgemm

#ifndef BS
#define BS 64
#endif

static_assert(BS > 0, "BS must be positive");

#if defined(__aarch64__)
#define MATMUL_HAVE_NEON 1
#endif
#if defined(__APPLE__)
#define MATMUL_HAVE_ACCELERATE 1
#endif

namespace calculate_matrix
{
    // Textbook i-j-k order: B is read down a column.
    void
    bad_matrix_multiply(double *const *C, const double *const *A, const double *const *B, int N);

    // i-k-j order: B and C are both read along rows.
    void
    fast_matrix_multiply(double *const *C, const double *const *A, const double *const *B, int N);

    //  Tiled, matrices as arrays of row pointers ---

    void matmul_tiled(double *const *C, const double *const *A, const double *const *B, int N);

    // Tiled, matrices as flat row-major arrays ---

    // Note the argument order of the two matmul_tiled_flat functions: inputs first.
    void matmul_tiled_flat(const double *A, const double *B, double *C, int N, int T);

    void matmul_tiled_flat_v2(const double *A, const double *B, double *C, int N, int T);

    void blocked_matrix_multiply(double *C, const double *A, const double *B, int N);

    //  Tiled and manually unrolled over k

    void blocked_k4_matrix_multiply(double *C, const double *A, const double *B, int N);

    void blocked_k4_ikj_matrix_multiply(double *C, const double *A, const double *B, int N);

    void blocked_matrix_multiply_k8(double *C, const double *A, const double *B, int N, int bs);

    // Same k4 kernel as blocked_k4_ikj_matrix_multiply, but with no tiling.
    void ikj_k4_matrix_multiply(double *C, const double *A, const double *B, int N);

#if defined(MATMUL_HAVE_NEON)
    // Tiles jj-kk with a packed copy of the B tile and a 4x4 NEON micro-kernel
    void neon_4x4_matrix_multiply(double *C, const double *A, const double *B, int N);
#endif

#if defined(MATMUL_HAVE_ACCELERATE)
    void accelerate_dgemm_matrix_multiply(double *C, const double *A, const double *B, int N);
#endif

    // GEMM: C (M x N) = A (M x K) * B (K x N), row-major
    void gemm_ikj(double *C, const double *A, const double *B, int M, int N, int K);
    void gemm_ikj(float *C, const float *A, const float *B, int M, int N, int K);

#if defined(MATMUL_HAVE_NEON)
    void gemm_neon_4x4(float *C, const float *A, const float *B, int M, int N, int K);
#endif

    // LU helper
    // A22 -= L21 * U12, where the panel is columns [k, k + kb). Tile size T.
    // Requires 0 <= k, 0 <= kb and k + kb <= n.
    void gemm_sub_tiled(double *A, int n, int k, int kb, int T);
} // namespace calculate_matrix