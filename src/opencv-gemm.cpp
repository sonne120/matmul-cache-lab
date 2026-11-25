#include "matrix-multiply.hpp"
#include <opencv2/core.hpp>

namespace calculate_matrix
{
    // C = op(A) * op(B) via cv::gemm (supports cv::GEMM_1_T / cv::GEMM_2_T)
    void opencv_gemm_f64(const cv::Mat &A, const cv::Mat &B, cv::Mat &C, int flags)
    {
        cv::gemm(A, B, 1.0, cv::Mat(), 0.0, C, flags);
    }

    void opencv_gemm_f32(const cv::Mat &A, const cv::Mat &B, cv::Mat &C, int flags)
    {
        cv::gemm(A, B, 1.0, cv::Mat(), 0.0, C, flags);
    }

    // C = alpha * op(A) * op(B) + beta * op(D)
    void opencv_gemm_axpb_f64(const cv::Mat &A, const cv::Mat &B, const cv::Mat &D, cv::Mat &C,
                              double alpha, double beta, int flags)
    {
        cv::gemm(A, B, alpha, D, beta, C, flags);
    }

    // Complex GEMM (2-channel CV_64FC2)
    void opencv_gemm_complex_f64(const cv::Mat &A, const cv::Mat &B, cv::Mat &C, int flags)
    {
        cv::gemm(A, B, 1.0, cv::Mat(), 0.0, C, flags);
    }

    // C = A * B via cv::MatExpr operator*
    void opencv_matmul_expr_f64(const cv::Mat &A, const cv::Mat &B, cv::Mat &C)
    {
        C = A * B;
    }

    void opencv_matmul_expr_f32(const cv::Mat &A, const cv::Mat &B, cv::Mat &C)
    {
        C = A * B;
    }

    // Gram matrix: C = scale * (A - delta)^T * (A - delta) [or A * A^T if aTa == false]
    void opencv_mul_transposed_f64(const cv::Mat &A, cv::Mat &C, bool aTa)
    {
        cv::mulTransposed(A, C, aTa);
    }

    void opencv_mul_transposed_f64(const cv::Mat &A, cv::Mat &C, bool aTa, const cv::Mat &delta,
                                   double scale)
    {
        cv::mulTransposed(A, C, aTa, delta, scale);
    }

    // Per-pixel multi-channel transform: dst(y,x) = M * src(y,x)
    void opencv_transform_f32(const cv::Mat &src, const cv::Mat &M, cv::Mat &dst)
    {
        cv::transform(src, dst, M);
    }

    // Fixed-size stack-allocated Matx multiply
    cv::Matx<float, 2, 4> opencv_matx_mul_2x3_3x4(const cv::Matx<float, 2, 3> &A,
                                                  const cv::Matx<float, 3, 4> &B)
    {
        return A * B;
    }
}
