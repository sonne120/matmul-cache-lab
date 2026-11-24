#pragma once

//
// Square multiply, C = A * B, all N x N
//
// Function                          Storage         Outer tile order  Tile size  k per pass
// --------------------------------  --------------  ----------------  ---------  ---------------
// bad_matrix_multiply               row pointers    none (i-j-k)      -          1
// fast_matrix_multiply              row pointers    none (i-k-j)      -          1
// matmul_tiled                      row pointers    ii-kk-jj          BS         1
// matmul_tiled_flat                 flat row-major  ii-kk-jj          argument   1
// matmul_tiled_flat_v2              flat row-major  ii-jj-kk          argument   1
// blocked_matrix_multiply           flat row-major  si-sk-sj          BS         1
// blocked_k4_matrix_multiply        flat row-major  sj-si-sk          BS         4
// blocked_k4_ikj_matrix_multiply    flat row-major  ii-kk-jj          BS         4
// blocked_matrix_multiply_k8        flat row-major  si-sj-sk          argument   8
// ikj_k4_matrix_multiply            flat row-major  none (i-k-j)      -          4
// neon_4x4_matrix_multiply          flat row-major  jj-kk, packed B   BS         4x4 NEON kernel
// accelerate_dgemm_matrix_multiply  flat row-major  Apple Accelerate cblas_dgemm
//
// Rectangular and helper kernels, flat row-major
//
// Function        Computes                                 Notes
// --------------  ---------------------------------------  ---------------------------------
// gemm_ikj        C (M x N) = A (M x K) * B (K x N)        double and float overloads
// gemm_neon_4x4   C (M x N) = A (M x K) * B (K x N)        float, 4x4 NEON kernel
// gemm_sub_tiled  A22 -= L21 * U12 in place, n x n matrix  LU helper; panel is columns
//                                                          [k, k + kb), tile size T; requires
//                                                          0 <= k, 0 <= kb, k + kb <= n

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
    void
    bad_matrix_multiply(double *const *C, const double *const *A, const double *const *B, int N);
    void
    fast_matrix_multiply(double *const *C, const double *const *A, const double *const *B, int N);
    void matmul_tiled(double *const *C, const double *const *A, const double *const *B, int N);

    void matmul_tiled_flat(const double *A, const double *B, double *C, int N, int T);
    void matmul_tiled_flat_v2(const double *A, const double *B, double *C, int N, int T);
    void blocked_matrix_multiply(double *C, const double *A, const double *B, int N);
    void blocked_k4_matrix_multiply(double *C, const double *A, const double *B, int N);
    void blocked_k4_ikj_matrix_multiply(double *C, const double *A, const double *B, int N);
    void blocked_matrix_multiply_k8(double *C, const double *A, const double *B, int N, int bs);
    void ikj_k4_matrix_multiply(double *C, const double *A, const double *B, int N);

#if defined(MATMUL_HAVE_NEON)
    void neon_4x4_matrix_multiply(double *C, const double *A, const double *B, int N);
#endif

#if defined(MATMUL_HAVE_ACCELERATE)
    void accelerate_dgemm_matrix_multiply(double *C, const double *A, const double *B, int N);
#endif

    void gemm_ikj(double *C, const double *A, const double *B, int M, int N, int K);
    void gemm_ikj(float *C, const float *A, const float *B, int M, int N, int K);

#if defined(MATMUL_HAVE_NEON)
    void gemm_neon_4x4(float *C, const float *A, const float *B, int M, int N, int K);
#endif

    void gemm_sub_tiled(double *A, int n, int k, int kb, int T);
}
