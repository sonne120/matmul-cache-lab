#include "matrix-multiply.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <vector>

#if defined(MATMUL_HAVE_NEON)
#include <arm_neon.h>
#endif

#define MATMUL_RESTRICT __restrict

namespace calculate_matrix
{
    namespace
    {
        template <typename T> T *row(T *matrix, int i, int n) noexcept
        {
            assert(matrix != nullptr);
            assert(i >= 0 && i < n);

            return matrix + static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
        }

        void zero_matrix(double *const *matrix, int n)
        {
            assert(n >= 0);

            for (int i = 0; i < n; ++i)
                std::fill_n(matrix[i], n, 0.0);
        }

        void zero_matrix(double *matrix, int n)
        {
            assert(n >= 0);

            std::fill_n(matrix, static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
        }
    }

    // Scalar order i-j-k.
    void bad_matrix_multiply(double *const *MATMUL_RESTRICT C,
                             const double *const *MATMUL_RESTRICT A,
                             const double *const *MATMUL_RESTRICT B,
                             int N)
    {
        assert(N >= 0);

        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                double sum = 0.0;

                // A is read along a row, B down a column.
                for (int k = 0; k < N; ++k)
                    sum += A[i][k] * B[k][j];

                C[i][j] = sum;
            }
        }
    }

    // Scalar order i-k-j.
    void fast_matrix_multiply(double *const *MATMUL_RESTRICT C,
                              const double *const *MATMUL_RESTRICT A,
                              const double *const *MATMUL_RESTRICT B,
                              int N)
    {
        assert(N >= 0);

        zero_matrix(C, N);

        for (int i = 0; i < N; ++i)
        {
            for (int k = 0; k < N; ++k)
            {
                // A[i][k] does not change in the inner loop: keep it in a register.
                const double a_ik = A[i][k];

                // Sequential access for C and B.
                for (int j = 0; j < N; ++j)
                    C[i][j] += a_ik * B[k][j];
            }
        }
    }

    // Outer tile order: ii-kk-jj.
    // Inner scalar order: i-k-j.
    void matmul_tiled(double *const *MATMUL_RESTRICT C,
                      const double *const *MATMUL_RESTRICT A,
                      const double *const *MATMUL_RESTRICT B,
                      int N)
    {
        assert(N >= 0);

        zero_matrix(C, N);

        for (int ii = 0; ii < N; ii += BS)
        {
            const int i_end = std::min(ii + BS, N);

            for (int kk = 0; kk < N; kk += BS)
            {
                const int k_end = std::min(kk + BS, N);

                for (int jj = 0; jj < N; jj += BS)
                {
                    const int j_end = std::min(jj + BS, N);

                    for (int i = ii; i < i_end; ++i)
                    {
                        // Row pointers are loaded once, outside the inner loop.
                        double *c_row = C[i];

                        for (int k = kk; k < k_end; ++k)
                        {
                            const double a_ik = A[i][k];
                            const double *b_row = B[k];

                            for (int j = jj; j < j_end; ++j)
                                c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }

    // Outer tile order: ii-kk-jj.
    // Inner scalar order: i-k-j.
    void matmul_tiled_flat(const double *MATMUL_RESTRICT A,
                           const double *MATMUL_RESTRICT B,
                           double *MATMUL_RESTRICT C,
                           int N,
                           int T)
    {
        assert(N >= 0);
        assert(T > 0);

        zero_matrix(C, N);

        for (int ii = 0; ii < N;)
        {
            const int i_end = ii + std::min(T, N - ii);

            for (int kk = 0; kk < N;)
            {
                const int k_end = kk + std::min(T, N - kk);

                for (int jj = 0; jj < N;)
                {
                    const int j_end = jj + std::min(T, N - jj);

                    for (int i = ii; i < i_end; ++i)
                    {
                        const double *a_row = row(A, i, N);
                        double *c_row = row(C, i, N);

                        for (int k = kk; k < k_end; ++k)
                        {
                            const double a_ik = a_row[k];
                            const double *b_row = row(B, k, N);

                            for (int j = jj; j < j_end; ++j)
                                c_row[j] += a_ik * b_row[j];
                        }
                    }

                    jj = j_end;
                }

                kk = k_end;
            }

            ii = i_end;
        }
    }

    // Outer tile order: ii-jj-kk.
    // Inner scalar order: i-k-j.
    void matmul_tiled_flat_v2(const double *A, const double *B, double *C, int N, int T)
    {
        assert(N >= 0);
        assert(T > 0);

        zero_matrix(C, N);

        for (int ii = 0; ii < N; ii += T)
        {
            const int i_end = std::min(ii + T, N);

            for (int jj = 0; jj < N; jj += T)
            {
                const int j_end = std::min(jj + T, N);

                for (int kk = 0; kk < N; kk += T)
                {
                    const int k_end = std::min(kk + T, N);

                    for (int i = ii; i < i_end; ++i)
                    {
                        const double *a_row = row(A, i, N);
                        double *c_row = row(C, i, N);

                        for (int k = kk; k < k_end; ++k)
                        {
                            const double a_ik = a_row[k];
                            const double *b_row = row(B, k, N);

                            for (int j = jj; j < j_end; ++j)
                                c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }

    // Basic tiled kernel.
    // Outer tile order: si-sk-sj.
    // Inner scalar order: i-k-j.
    void blocked_matrix_multiply(double *MATMUL_RESTRICT C,
                                 const double *MATMUL_RESTRICT A,
                                 const double *MATMUL_RESTRICT B,
                                 int N)
    {
        assert(N >= 0);

        zero_matrix(C, N);

        for (int si = 0; si < N; si += BS)
        {
            const int i_end = std::min(si + BS, N);

            for (int sk = 0; sk < N; sk += BS)
            {
                const int k_end = std::min(sk + BS, N);

                for (int sj = 0; sj < N; sj += BS)
                {
                    const int j_end = std::min(sj + BS, N);

                    for (int i = si; i < i_end; ++i)
                    {
                        const double *a_row = row(A, i, N);
                        double *c_row = row(C, i, N);

                        for (int k = sk; k < k_end; ++k)
                        {
                            const double a_ik = a_row[k];
                            const double *b_row = row(B, k, N);

                            for (int j = sj; j < j_end; ++j)
                                c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }

    // K-unrolled-by-4 kernel with outer tile order sj-si-sk.
    void blocked_k4_matrix_multiply(double *MATMUL_RESTRICT C,
                                    const double *MATMUL_RESTRICT A,
                                    const double *MATMUL_RESTRICT B,
                                    int N)
    {
        assert(N >= 0);

        zero_matrix(C, N);

        for (int sj = 0; sj < N; sj += BS)
        {
            const int j_end = std::min(sj + BS, N);

            for (int si = 0; si < N; si += BS)
            {
                const int i_end = std::min(si + BS, N);

                for (int sk = 0; sk < N; sk += BS)
                {
                    const int k_end = std::min(sk + BS, N);

                    for (int i = si; i < i_end; ++i)
                    {
                        const double *a_row = row(A, i, N);
                        double *c_row = row(C, i, N);

                        int k = sk;

                        // Four rows of B per pass.
                        for (; k + 4 <= k_end; k += 4)
                        {
                            const double a0 = a_row[k];
                            const double a1 = a_row[k + 1];
                            const double a2 = a_row[k + 2];
                            const double a3 = a_row[k + 3];

                            const double *b0 = row(B, k, N);
                            const double *b1 = b0 + N;
                            const double *b2 = b1 + N;
                            const double *b3 = b2 + N;

                            for (int j = sj; j < j_end; ++j)
                                c_row[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
                        }

                        // Remaining 0..3 rows, one at a time.
                        for (; k < k_end; ++k)
                        {
                            const double a_ik = a_row[k];
                            const double *b_row = row(B, k, N);

                            for (int j = sj; j < j_end; ++j)
                                c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }

    // K-unrolled-by-4 kernel with outer tile order ii-kk-jj.
    void blocked_k4_ikj_matrix_multiply(double *MATMUL_RESTRICT C,
                                        const double *MATMUL_RESTRICT A,
                                        const double *MATMUL_RESTRICT B,
                                        int N)
    {
        assert(N >= 0);

        zero_matrix(C, N);

        for (int ii = 0; ii < N; ii += BS)
        {
            const int i_end = std::min(ii + BS, N);

            for (int kk = 0; kk < N; kk += BS)
            {
                const int k_end = std::min(kk + BS, N);

                for (int jj = 0; jj < N; jj += BS)
                {
                    const int j_end = std::min(jj + BS, N);

                    for (int i = ii; i < i_end; ++i)
                    {
                        const double *a_row = row(A, i, N);
                        double *c_row = row(C, i, N);

                        int k = kk;

                        for (; k + 4 <= k_end; k += 4)
                        {
                            const double a0 = a_row[k];
                            const double a1 = a_row[k + 1];
                            const double a2 = a_row[k + 2];
                            const double a3 = a_row[k + 3];

                            const double *b0 = row(B, k, N);
                            const double *b1 = b0 + N;
                            const double *b2 = b1 + N;
                            const double *b3 = b2 + N;

                            for (int j = jj; j < j_end; ++j)
                                c_row[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
                        }

                        // Remaining 0..3 rows, one at a time.
                        for (; k < k_end; ++k)
                        {
                            const double a_ik = a_row[k];
                            const double *b_row = row(B, k, N);

                            for (int j = jj; j < j_end; ++j)
                                c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }

    // K-unrolled-by-4 kernel with no tiling: plain i-k-j over the whole matrix.
    void ikj_k4_matrix_multiply(double *MATMUL_RESTRICT C,
                                const double *MATMUL_RESTRICT A,
                                const double *MATMUL_RESTRICT B,
                                int N)
    {
        assert(N >= 0);

        zero_matrix(C, N);

        for (int i = 0; i < N; ++i)
        {
            const double *a_row = row(A, i, N);
            double *c_row = row(C, i, N);

            int k = 0;

            // Four rows of B per pass.
            for (; k + 4 <= N; k += 4)
            {
                const double a0 = a_row[k];
                const double a1 = a_row[k + 1];
                const double a2 = a_row[k + 2];
                const double a3 = a_row[k + 3];

                const double *b0 = row(B, k, N);
                const double *b1 = b0 + N;
                const double *b2 = b1 + N;
                const double *b3 = b2 + N;

                for (int j = 0; j < N; ++j)
                    c_row[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
            }

            // Remaining 0..3 rows, one at a time.
            for (; k < N; ++k)
            {
                const double a_ik = a_row[k];
                const double *b_row = row(B, k, N);

                for (int j = 0; j < N; ++j)
                    c_row[j] += a_ik * b_row[j];
            }
        }
    }

    // K-unrolled-by-8 kernel with outer tile order si-sj-sk.
    void blocked_matrix_multiply_k8(double *MATMUL_RESTRICT C,
                                    const double *MATMUL_RESTRICT A,
                                    const double *MATMUL_RESTRICT B,
                                    int N,
                                    int bs)
    {
        assert(N >= 0);
        assert(bs > 0);

        zero_matrix(C, N);

        for (int si = 0; si < N; si += bs)
        {
            const int i_end = std::min(si + bs, N);

            for (int sj = 0; sj < N; sj += bs)
            {
                const int j_end = std::min(sj + bs, N);

                for (int sk = 0; sk < N; sk += bs)
                {
                    const int k_end = std::min(sk + bs, N);

                    for (int i = si; i < i_end; ++i)
                    {
                        const double *a_row = row(A, i, N);
                        double *c_row = row(C, i, N);

                        int k = sk;

                        for (; k + 8 <= k_end; k += 8)
                        {
                            const double a0 = a_row[k];
                            const double a1 = a_row[k + 1];
                            const double a2 = a_row[k + 2];
                            const double a3 = a_row[k + 3];
                            const double a4 = a_row[k + 4];
                            const double a5 = a_row[k + 5];
                            const double a6 = a_row[k + 6];
                            const double a7 = a_row[k + 7];

                            const double *b0 = row(B, k, N);
                            const double *b1 = b0 + N;
                            const double *b2 = b1 + N;
                            const double *b3 = b2 + N;
                            const double *b4 = b3 + N;
                            const double *b5 = b4 + N;
                            const double *b6 = b5 + N;
                            const double *b7 = b6 + N;

                            for (int j = sj; j < j_end; ++j)
                                c_row[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j]
                                            + a4 * b4[j] + a5 * b5[j] + a6 * b6[j] + a7 * b7[j];
                        }
                        for (; k + 4 <= k_end; k += 4)
                        {
                            const double a0 = a_row[k];
                            const double a1 = a_row[k + 1];
                            const double a2 = a_row[k + 2];
                            const double a3 = a_row[k + 3];

                            const double *b0 = row(B, k, N);
                            const double *b1 = b0 + N;
                            const double *b2 = b1 + N;
                            const double *b3 = b2 + N;

                            for (int j = sj; j < j_end; ++j)
                                c_row[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
                        }

                        for (; k < k_end; ++k)
                        {
                            const double a_ik = a_row[k];
                            const double *b_row = row(B, k, N);

                            for (int j = sj; j < j_end; ++j)
                                c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }

    // The panel [k, k + kb) splits the matrix into four blocks:
    //
    //     +---------+---------+
    //     |   LU    |   U12   |
    //     +---------+---------+
    //     |   L21   |   A22   |
    //     +---------+---------+
    //
    // The update A22 -= L21 * U12 is a matrix multiplication done in place.
    // Outer tile order: ii-pp-jj.
    // Inner scalar order: i-p-j.
    void gemm_sub_tiled(double *A, int n, int k, int kb, int T)
    {
        assert(n >= 0);
        assert(T > 0);
        assert(k >= 0);
        assert(k <= n);
        assert(kb >= 0);
        assert(kb <= n - k);

        const int tail = k + kb; // first row and column of A22

        for (int ii = tail; ii < n; ii += T)
        {
            const int i_end = std::min(ii + T, n);

            // Tiles along the panel width.
            for (int pp = k; pp < tail; pp += T)
            {
                const int p_end = std::min(pp + T, tail);

                for (int jj = tail; jj < n; jj += T)
                {
                    const int j_end = std::min(jj + T, n);

                    for (int i = ii; i < i_end; ++i)
                    {
                        double *a_row = row(A, i, n);

                        for (int p = pp; p < p_end; ++p)
                        {
                            const double l_ip = a_row[p];       // L21(i, p)
                            const double *u_row = row(A, p, n); // row of U12

                            for (int j = jj; j < j_end; ++j)
                                a_row[j] -= l_ip * u_row[j];
                        }
                    }
                }
            }
        }
    }

#if defined(MATMUL_HAVE_NEON)
    // 4x4 NEON micro-kernel: a 4x4 block of C is kept in eight registers while
    // kb rows of the packed B strip are folded into it.
    static inline void micro_kernel_4x4(double *MATMUL_RESTRICT c0,
                                        double *MATMUL_RESTRICT c1,
                                        double *MATMUL_RESTRICT c2,
                                        double *MATMUL_RESTRICT c3,
                                        const double *MATMUL_RESTRICT a0,
                                        const double *MATMUL_RESTRICT a1,
                                        const double *MATMUL_RESTRICT a2,
                                        const double *MATMUL_RESTRICT a3,
                                        const double *MATMUL_RESTRICT bp,
                                        int kb)
    {
        float64x2_t c00 = vld1q_f64(c0), c01 = vld1q_f64(c0 + 2);
        float64x2_t c10 = vld1q_f64(c1), c11 = vld1q_f64(c1 + 2);
        float64x2_t c20 = vld1q_f64(c2), c21 = vld1q_f64(c2 + 2);
        float64x2_t c30 = vld1q_f64(c3), c31 = vld1q_f64(c3 + 2);

        for (int k = 0; k < kb; ++k)
        {
            const float64x2_t b0 = vld1q_f64(bp);     // B[k][j], B[k][j+1]
            const float64x2_t b1 = vld1q_f64(bp + 2); // B[k][j+2], B[k][j+3]
            bp += 4;

            c00 = vfmaq_n_f64(c00, b0, a0[k]);
            c01 = vfmaq_n_f64(c01, b1, a0[k]);
            c10 = vfmaq_n_f64(c10, b0, a1[k]);
            c11 = vfmaq_n_f64(c11, b1, a1[k]);
            c20 = vfmaq_n_f64(c20, b0, a2[k]);
            c21 = vfmaq_n_f64(c21, b1, a2[k]);
            c30 = vfmaq_n_f64(c30, b0, a3[k]);
            c31 = vfmaq_n_f64(c31, b1, a3[k]);
        }

        vst1q_f64(c0, c00);
        vst1q_f64(c0 + 2, c01);
        vst1q_f64(c1, c10);
        vst1q_f64(c1 + 2, c11);
        vst1q_f64(c2, c20);
        vst1q_f64(c2 + 2, c21);
        vst1q_f64(c3, c30);
        vst1q_f64(c3 + 2, c31);
    }

    // Outer tile order: jj-kk, then four rows of C at a time.
    // The B tile is copied into a contiguous buffer, one 4-column strip at a time.
    void neon_4x4_matrix_multiply(double *MATMUL_RESTRICT C,
                                  const double *MATMUL_RESTRICT A,
                                  const double *MATMUL_RESTRICT B,
                                  int N)
    {
        assert(N >= 0);

        const std::size_t n = static_cast<std::size_t>(N);
        zero_matrix(C, N);

        // Packed copy of one B tile: for every 4-column strip, kb rows of 4 doubles, contiguous.
        std::vector<double> packed(static_cast<std::size_t>(BS) * BS);

        // const int i4 = N & ~3; // rows covered by the 4x4 kernel
        const int i4 = N - N % 4; // instead of ~3

        for (int jj = 0; jj < N; jj += BS)
        {
            const int jb = std::min(BS, N - jj);
            // const int j4 = jb & ~3; // columns  tile covered by the 4x4 kernel
            const int j4 = jb - jb % 4;

            for (int kk = 0; kk < N; kk += BS)
            {
                const int kb = std::min(BS, N - kk);

                // Pack B[kk..kk+kb) x [jj..jj+j4).
                double *dst = packed.data();
                for (int js = 0; js < j4; js += 4)
                {
                    for (int k = 0; k < kb; ++k)
                    {
                        const double *src = B + static_cast<std::size_t>(kk + k) * n + (jj + js);
                        dst[0] = src[0];
                        dst[1] = src[1];
                        dst[2] = src[2];
                        dst[3] = src[3];
                        dst += 4;
                    }
                }

                for (int i = 0; i < i4; i += 4)
                {
                    double *c0 = C + static_cast<std::size_t>(i) * n + jj;
                    double *c1 = c0 + n;
                    double *c2 = c1 + n;
                    double *c3 = c2 + n;
                    const double *a0 = A + static_cast<std::size_t>(i) * n + kk;
                    const double *a1 = a0 + n;
                    const double *a2 = a1 + n;
                    const double *a3 = a2 + n;

                    const double *bp = packed.data();
                    for (int js = 0; js < j4; js += 4)
                    {
                        micro_kernel_4x4(c0 + js, c1 + js, c2 + js, c3 + js, a0, a1, a2, a3, bp,
                                         kb);
                        bp += static_cast<std::size_t>(kb) * 4;
                    }

                    // Leftover columns of the tile (jb not a multiple of 4).
                    for (int j = j4; j < jb; ++j)
                    {
                        double s0 = c0[j], s1 = c1[j], s2 = c2[j], s3 = c3[j];
                        for (int k = 0; k < kb; ++k)
                        {
                            const double b = B[static_cast<std::size_t>(kk + k) * n + (jj + j)];
                            s0 += a0[k] * b;
                            s1 += a1[k] * b;
                            s2 += a2[k] * b;
                            s3 += a3[k] * b;
                        }
                        c0[j] = s0;
                        c1[j] = s1;
                        c2[j] = s2;
                        c3[j] = s3;
                    }
                }

                // Leftover rows (N not a multiple of 4).
                for (int i = i4; i < N; ++i)
                {
                    double *c = C + static_cast<std::size_t>(i) * n + jj;
                    const double *a = A + static_cast<std::size_t>(i) * n + kk;
                    for (int k = 0; k < kb; ++k)
                    {
                        const double aik = a[k];
                        const double *b = B + static_cast<std::size_t>(kk + k) * n + jj;
                        for (int j = 0; j < jb; ++j)
                            c[j] += aik * b[j];
                    }
                }
            }
        }
    }
#endif

    namespace
    {
        template <typename T> void gemm_ikj_impl(T *C, const T *A, const T *B, int M, int N, int K)
        {
            std::fill_n(C, static_cast<std::size_t>(M) * N, T(0));
            for (int i = 0; i < M; ++i)
            {
                T *c_row = C + static_cast<std::size_t>(i) * N;
                const T *a_row = A + static_cast<std::size_t>(i) * K;
                for (int k = 0; k < K; ++k)
                {
                    const T a_ik = a_row[k];
                    const T *b_row = B + static_cast<std::size_t>(k) * N;
                    for (int j = 0; j < N; ++j)
                        c_row[j] += a_ik * b_row[j];
                }
            }
        }
    }

    void gemm_ikj(double *C, const double *A, const double *B, int M, int N, int K)
    {
        gemm_ikj_impl(C, A, B, M, N, K);
    }

    void gemm_ikj(float *C, const float *A, const float *B, int M, int N, int K)
    {
        gemm_ikj_impl(C, A, B, M, N, K);
    }

#if defined(MATMUL_HAVE_NEON)
    void gemm_neon_4x4(float *C, const float *A, const float *B, int M, int N, int K)
    {
        std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0f);

        int i = 0;
        for (; i + 4 <= M; i += 4)
        {
            float *c0 = C + static_cast<std::size_t>(i + 0) * N;
            float *c1 = C + static_cast<std::size_t>(i + 1) * N;
            float *c2 = C + static_cast<std::size_t>(i + 2) * N;
            float *c3 = C + static_cast<std::size_t>(i + 3) * N;

            const float *a0 = A + static_cast<std::size_t>(i + 0) * K;
            const float *a1 = A + static_cast<std::size_t>(i + 1) * K;
            const float *a2 = A + static_cast<std::size_t>(i + 2) * K;
            const float *a3 = A + static_cast<std::size_t>(i + 3) * K;

            int j = 0;
            for (; j + 4 <= N; j += 4)
            {
                float32x4_t acc0 = vdupq_n_f32(0.0f);
                float32x4_t acc1 = vdupq_n_f32(0.0f);
                float32x4_t acc2 = vdupq_n_f32(0.0f);
                float32x4_t acc3 = vdupq_n_f32(0.0f);

                for (int k = 0; k < K; ++k)
                {
                    const float32x4_t b_vec = vld1q_f32(B + static_cast<std::size_t>(k) * N + j);
                    acc0 = vfmaq_n_f32(acc0, b_vec, a0[k]);
                    acc1 = vfmaq_n_f32(acc1, b_vec, a1[k]);
                    acc2 = vfmaq_n_f32(acc2, b_vec, a2[k]);
                    acc3 = vfmaq_n_f32(acc3, b_vec, a3[k]);
                }

                vst1q_f32(c0 + j, acc0);
                vst1q_f32(c1 + j, acc1);
                vst1q_f32(c2 + j, acc2);
                vst1q_f32(c3 + j, acc3);
            }

            for (; j < N; ++j)
            {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    const float b_kj = B[static_cast<std::size_t>(k) * N + j];
                    s0 += a0[k] * b_kj;
                    s1 += a1[k] * b_kj;
                    s2 += a2[k] * b_kj;
                    s3 += a3[k] * b_kj;
                }
                c0[j] = s0;
                c1[j] = s1;
                c2[j] = s2;
                c3[j] = s3;
            }
        }

        for (; i < M; ++i)
        {
            float *c_row = C + static_cast<std::size_t>(i) * N;
            const float *a_row = A + static_cast<std::size_t>(i) * K;
            for (int k = 0; k < K; ++k)
            {
                const float a_ik = a_row[k];
                const float *b_row = B + static_cast<std::size_t>(k) * N;
                for (int j = 0; j < N; ++j)
                    c_row[j] += a_ik * b_row[j];
            }
        }
    }
#endif

    void conv2d_naive_f32(const float *src, int H, int W, const float *kernel, int K, float *dst)
    {
        const int pad = K / 2;
        for (int y = 0; y < H; ++y)
        {
            float *dst_row = dst + static_cast<std::size_t>(y) * W;
            for (int x = 0; x < W; ++x)
            {
                float sum = 0.0f;
                for (int ky = 0; ky < K; ++ky)
                {
                    const int sy = y + ky - pad;
                    if (sy < 0 || sy >= H)
                        continue;
                    const float *src_row = src + static_cast<std::size_t>(sy) * W;
                    const float *ker_row = kernel + static_cast<std::size_t>(ky) * K;
                    for (int kx = 0; kx < K; ++kx)
                    {
                        const int sx = x + kx - pad;
                        if (sx >= 0 && sx < W)
                            sum += src_row[sx] * ker_row[kx];
                    }
                }
                dst_row[x] = sum;
            }
        }
    }

    void im2col_f32(const float *src, int H, int W, int K, float *col)
    {
        const int pad = K / 2;
        const int kk = K * K;
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                float *dst_patch = col + (static_cast<std::size_t>(y) * W + x) * kk;
                int idx = 0;
                for (int ky = 0; ky < K; ++ky)
                {
                    const int sy = y + ky - pad;
                    if (sy < 0 || sy >= H)
                    {
                        for (int kx = 0; kx < K; ++kx)
                            dst_patch[idx++] = 0.0f;
                        continue;
                    }
                    const float *src_row = src + static_cast<std::size_t>(sy) * W;
                    for (int kx = 0; kx < K; ++kx)
                    {
                        const int sx = x + kx - pad;
                        dst_patch[idx++] = (sx >= 0 && sx < W) ? src_row[sx] : 0.0f;
                    }
                }
            }
        }
    }

    void color_transform_gemm_f32(const float *src, int H, int W, const float *M3x3, float *dst)
    {
        for (int y = 0; y < H; ++y)
        {
            const float *s = src + static_cast<std::size_t>(y) * W * 3;
            float *d = dst + static_cast<std::size_t>(y) * W * 3;
            for (int x = 0; x < W; ++x)
            {
                const float r = s[3 * x + 0];
                const float g = s[3 * x + 1];
                const float b = s[3 * x + 2];
                d[3 * x + 0] = r * M3x3[0] + g * M3x3[1] + b * M3x3[2];
                d[3 * x + 1] = r * M3x3[3] + g * M3x3[4] + b * M3x3[5];
                d[3 * x + 2] = r * M3x3[6] + g * M3x3[7] + b * M3x3[8];
            }
        }
    }
}
