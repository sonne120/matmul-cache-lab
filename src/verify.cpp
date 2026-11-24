#include "verify.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <format>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "matrix-multiply.hpp"

namespace calculate_matrix
{
    namespace
    {
        constexpr std::array kSizes{250, 500, 1000, 2000};

        constexpr int kRectM = 127;
        constexpr int kRectN = 53;
        constexpr int kRectK = 37;

        constexpr double kPoison = 123.0;

        constexpr double kTolDouble = 1e-9;
        constexpr double kTolFloat = 1e-5;

        // A square kernel computes C = A * B for N x N row-major matrices.
        template <typename Fn>
        concept SquareKernel = std::invocable<Fn, double *, const double *, const double *, int>;

        [[nodiscard]] bool report(const std::string &label, double max_rel_err, double tolerance)
        {
            const bool ok = max_rel_err < tolerance;
            std::printf("%s max_rel_err=%.3g %s\n", label.c_str(), max_rel_err, ok ? "OK" : "FAIL");
            return ok;
        }

        template <std::floating_point T>
        [[nodiscard]] double max_rel_err(const std::vector<T> &got, const std::vector<T> &ref)
        {
            double worst = 0.0;
            for (std::size_t i = 0; i < got.size(); ++i)
                worst = std::max(worst, static_cast<double>(std::fabs(got[i] - ref[i]))
                                            / (1.0 + std::fabs(ref[i])));
            return worst;
        }

        // Random inputs for one matrix size together with the reference product A * B.
        struct SquareCase
        {
            int N;
            std::vector<double> A, B, expected;
        };

        [[nodiscard]] SquareCase make_square_case(int N)
        {
            const std::size_t n = static_cast<std::size_t>(N);
            SquareCase sc{.N = N,
                          .A = std::vector<double>(n * n),
                          .B = std::vector<double>(n * n),
                          .expected = std::vector<double>(n * n, 0.0)};

            std::mt19937_64 gen(42);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (auto &x : sc.A)
                x = dist(gen);
            for (auto &x : sc.B)
                x = dist(gen);

            // plain ikj.
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t k = 0; k < n; ++k)
                {
                    const double a = sc.A[i * n + k];
                    for (std::size_t j = 0; j < n; ++j)
                        sc.expected[i * n + j] += a * sc.B[k * n + j];
                }
            return sc;
        }

        // Runs one kernel on the case and reports whether it matches the reference.
        template <SquareKernel Fn>
        [[nodiscard]] bool check(const SquareCase &sc, std::string_view name, Fn kernel)
        {
            std::vector<double> C(sc.expected.size(), kPoison);
            kernel(C.data(), sc.A.data(), sc.B.data(), sc.N);

            return report(std::format("N={:<4} BS={:<3} {:<32}", sc.N, BS, name),
                          max_rel_err(C, sc.expected), kTolDouble);
        }

        [[nodiscard]] int verify_square(int N)
        {
            const SquareCase sc = make_square_case(N);
            int failures = 0;
            const auto run = [&](std::string_view name, SquareKernel auto kernel)
            {
                if (!check(sc, name, kernel))
                    ++failures;
            };

            run("matmul_tiled_flat", [](double *C, const double *A, const double *B, int n)
                { matmul_tiled_flat(A, B, C, n, BS); });
            run("matmul_tiled_flat_v2", [](double *C, const double *A, const double *B, int n)
                { matmul_tiled_flat_v2(A, B, C, n, BS); });
            run("blocked_matrix_multiply", blocked_matrix_multiply);
            run("blocked_k4_matrix_multiply", blocked_k4_matrix_multiply);
            run("blocked_k4_ikj_matrix_multiply", blocked_k4_ikj_matrix_multiply);
            run("blocked_matrix_multiply_k8", [](double *C, const double *A, const double *B, int n)
                { blocked_matrix_multiply_k8(C, A, B, n, BS); });
            run("ikj_k4_matrix_multiply", ikj_k4_matrix_multiply);
#if defined(MATMUL_HAVE_NEON)
            run("neon_4x4_matrix_multiply", neon_4x4_matrix_multiply);
#endif
#if defined(MATMUL_HAVE_ACCELERATE)
            run("accelerate_dgemm_matrix_multiply", accelerate_dgemm_matrix_multiply);
#endif
            return failures;
        }
    }

    int verify()
    {
        int failures = 0;

        for (const int N : kSizes)
            failures += verify_square(N);

        // Rectangular GEMM checks (M=127, N=53, K=37)
        {
            constexpr int M = kRectM, N = kRectN, K = kRectK;
            const auto rect_label = [&](std::string_view kernel)
            { return std::format("rect {}x{}x{} {:<26}", M, K, N, kernel); };
            std::mt19937_64 gen(7);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            const std::size_t mk = static_cast<std::size_t>(M) * K;
            const std::size_t kn = static_cast<std::size_t>(K) * N;
            const std::size_t mn = static_cast<std::size_t>(M) * N;

            std::vector<double> Ad(mk), Bd(kn), Cd(mn, kPoison), Rd(mn, 0.0);
            std::vector<float> Af(mk), Bf(kn), Cf(mn, static_cast<float>(kPoison)), Rf(mn, 0.0f);
            for (int i = 0; i < M * K; ++i)
            {
                Ad[i] = dist(gen);
                Af[i] = static_cast<float>(Ad[i]);
            }
            for (int i = 0; i < K * N; ++i)
            {
                Bd[i] = dist(gen);
                Bf[i] = static_cast<float>(Bd[i]);
            }
            for (int i = 0; i < M; ++i)
                for (int k = 0; k < K; ++k)
                    for (int j = 0; j < N; ++j)
                    {
                        Rd[i * N + j] += Ad[i * K + k] * Bd[k * N + j];
                        Rf[i * N + j] += Af[i * K + k] * Bf[k * N + j];
                    }

            gemm_ikj(Cd.data(), Ad.data(), Bd.data(), M, N, K);
            if (!report(rect_label("gemm_ikj(double)"), max_rel_err(Cd, Rd), kTolDouble))
                ++failures;

            gemm_ikj(Cf.data(), Af.data(), Bf.data(), M, N, K);
            if (!report(rect_label("gemm_ikj(float)"), max_rel_err(Cf, Rf), kTolFloat))
                ++failures;

#if defined(MATMUL_HAVE_NEON)
            std::fill(Cf.begin(), Cf.end(), static_cast<float>(kPoison));
            gemm_neon_4x4(Cf.data(), Af.data(), Bf.data(), M, N, K);
            if (!report(rect_label("gemm_neon_4x4(float)"), max_rel_err(Cf, Rf), kTolFloat))
                ++failures;
#endif
        }

        return failures;
    }
}
