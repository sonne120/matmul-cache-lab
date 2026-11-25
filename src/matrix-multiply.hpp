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
//
// OpenCV variants (src/opencv-gemm.cpp, enabled when MATMUL_HAVE_OPENCV is defined)
//
// Function                   Computes                                        Notes
// -------------------------  ----------------------------------------------  ----------------------------------------
// opencv_gemm_f64            C (M x N) = op(A) * op(B)                       CV_64F; supports GEMM_1_T / GEMM_2_T
// opencv_gemm_f32            C (M x N) = op(A) * op(B)                       CV_32F; supports GEMM_1_T / GEMM_2_T
// opencv_gemm_axpb_f64       C = alpha * op(A) * op(B) + beta * op(D)        CV_64F; full BLAS-style GEMM
// opencv_gemm_complex_f64    C (M x N) = op(A) * op(B)                       CV_64FC2 (2-channel complex)
// opencv_matmul_expr_f64     C (M x N) = A (M x K) * B (K x N)               CV_64F via MatExpr operator*
// opencv_matmul_expr_f32     C (M x N) = A (M x K) * B (K x N)               CV_32F via MatExpr operator*
// opencv_mul_transposed_f64  C = scale * (A - delta)^T * (A - delta)         Gram matrix; or A * A^T if aTa = false
// opencv_transform_f32       dst(y,x) = M * src(y,x)                         Per-pixel transform on H x W image
// opencv_matx_mul_2x3_3x4    C (2 x 4) = A (2 x 3) * B (3 x 4)               Fixed-size stack-allocated cv::Matx

#ifndef BS
#define BS 64
#endif

static_assert(BS > 0, "BS must be positive");

#if defined(__aarch64__)
#define MATMUL_HAVE_NEON 1
#endif
#if defined(__APPLE__) && defined(__clang__)
#define MATMUL_HAVE_ACCELERATE 1
#endif

#if defined(MATMUL_HAVE_OPENCV)
namespace cv
{
    class Mat;
    template <typename _Tp, int m, int n> class Matx;
}
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

#if defined(MATMUL_HAVE_ACCELERATE) || defined(MATMUL_HAVE_CBLAS)
    void accelerate_dgemm_matrix_multiply(double *C, const double *A, const double *B, int N);
#endif

    void gemm_ikj(double *C, const double *A, const double *B, int M, int N, int K);
    void gemm_ikj(float *C, const float *A, const float *B, int M, int N, int K);

#if defined(MATMUL_HAVE_NEON)
    void gemm_neon_4x4(float *C, const float *A, const float *B, int M, int N, int K);
#endif

    void conv2d_naive_f32(const float *src, int H, int W, const float *kernel, int K, float *dst);
    void im2col_f32(const float *src, int H, int W, int K, float *col);
    void color_transform_gemm_f32(const float *src, int H, int W, const float *M3x3, float *dst);

    void gemm_sub_tiled(double *A, int n, int k, int kb, int T);

#if defined(MATMUL_HAVE_OPENCV)
    void opencv_gemm_f64(const cv::Mat &A, const cv::Mat &B, cv::Mat &C, int flags = 0);
    void opencv_gemm_f32(const cv::Mat &A, const cv::Mat &B, cv::Mat &C, int flags = 0);
    void opencv_gemm_axpb_f64(const cv::Mat &A, const cv::Mat &B, const cv::Mat &D, cv::Mat &C,
                              double alpha = 1.0, double beta = 1.0, int flags = 0);
    void opencv_gemm_complex_f64(const cv::Mat &A, const cv::Mat &B, cv::Mat &C, int flags = 0);
    void opencv_matmul_expr_f64(const cv::Mat &A, const cv::Mat &B, cv::Mat &C);
    void opencv_matmul_expr_f32(const cv::Mat &A, const cv::Mat &B, cv::Mat &C);
    void opencv_mul_transposed_f64(const cv::Mat &A, cv::Mat &C, bool aTa = true);
    void opencv_mul_transposed_f64(const cv::Mat &A, cv::Mat &C, bool aTa, const cv::Mat &delta,
                                   double scale = 1.0);
    void opencv_transform_f32(const cv::Mat &src, const cv::Mat &M, cv::Mat &dst);
    cv::Matx<float, 2, 4> opencv_matx_mul_2x3_3x4(const cv::Matx<float, 2, 3> &A,
                                                  const cv::Matx<float, 3, 4> &B);
#endif
}
