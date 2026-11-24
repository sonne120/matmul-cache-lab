#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "matrix-multiply.hpp"

using namespace calculate_matrix;

namespace
{
    using FlatFn = void(double *C, const double *A, const double *B, int N);

    struct Candidate
    {
        const char *name;
        FlatFn *run;
    };

    const Candidate candidates[] = {
        {"matmul_tiled_flat", [](double *C, const double *A, const double *B, int N)
         { matmul_tiled_flat(A, B, C, N, BS); }},
        {"matmul_tiled_flat_v2", [](double *C, const double *A, const double *B, int N)
         { matmul_tiled_flat_v2(A, B, C, N, BS); }},
        {"blocked_matrix_multiply", blocked_matrix_multiply},
        {"blocked_k4_matrix_multiply", blocked_k4_matrix_multiply},
        {"blocked_k4_ikj_matrix_multiply", blocked_k4_ikj_matrix_multiply},
        {"blocked_matrix_multiply_k8", [](double *C, const double *A, const double *B, int N)
         { blocked_matrix_multiply_k8(C, A, B, N, BS); }},
        {"ikj_k4_matrix_multiply", ikj_k4_matrix_multiply},
#if defined(MATMUL_HAVE_NEON)
        {"neon_4x4_matrix_multiply", neon_4x4_matrix_multiply},
#endif
#if defined(MATMUL_HAVE_ACCELERATE)
        {"accelerate_dgemm_matrix_multiply", accelerate_dgemm_matrix_multiply},
#endif
    };

    int verify()
    {
        int failures = 0;

        for (int N : {250, 500, 1000, 2000})
        {
            const std::size_t n = static_cast<std::size_t>(N);
            std::vector<double> A(n * n), B(n * n), C(n * n), R(n * n, 0.0);

            std::mt19937_64 gen(42);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (auto &x : A)
                x = dist(gen);
            for (auto &x : B)
                x = dist(gen);

            // plain ikj.
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t k = 0; k < n; ++k)
                {
                    const double a = A[i * n + k];
                    for (std::size_t j = 0; j < n; ++j)
                        R[i * n + j] += a * B[k * n + j];
                }

            for (const Candidate &candidate : candidates)
            {
                // Poison C so that a function that forgets to zero it is caught.
                std::fill(C.begin(), C.end(), 123.0);
                candidate.run(C.data(), A.data(), B.data(), N);

                double max_err = 0.0;
                for (std::size_t idx = 0; idx < n * n; ++idx)
                    max_err =
                        std::max(max_err, std::fabs(C[idx] - R[idx]) / (1.0 + std::fabs(R[idx])));

                const bool ok = max_err < 1e-9;
                std::printf("N=%-4d BS=%-3d %-32s max_rel_err=%.3g %s\n", N, BS, candidate.name,
                            max_err, ok ? "OK" : "FAIL");
                failures += !ok;
            }
        }

        // Rectangular GEMM checks (M=127, N=53, K=37)
        {
            const int M = 127, N = 53, K = 37;
            std::mt19937_64 gen(7);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            const std::size_t mk = static_cast<std::size_t>(M) * K;
            const std::size_t kn = static_cast<std::size_t>(K) * N;
            const std::size_t mn = static_cast<std::size_t>(M) * N;

            std::vector<double> Ad(mk), Bd(kn), Cd(mn, 123.0), Rd(mn, 0.0);
            std::vector<float> Af(mk), Bf(kn), Cf(mn, 123.0), Rf(mn, 0.0f);
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
            double err_d = 0.0;
            for (int i = 0; i < M * N; ++i)
                err_d = std::max(err_d, std::fabs(Cd[i] - Rd[i]) / (1.0 + std::fabs(Rd[i])));
            const bool ok_d = err_d < 1e-9;
            std::printf("rect 127x37x53 %-26s max_rel_err=%.3g %s\n", "gemm_ikj(double)", err_d,
                        ok_d ? "OK" : "FAIL");
            failures += !ok_d;

            gemm_ikj(Cf.data(), Af.data(), Bf.data(), M, N, K);
            double err_f = 0.0;
            for (int i = 0; i < M * N; ++i)
                err_f = std::max(err_f, static_cast<double>(std::fabs(Cf[i] - Rf[i]))
                                            / (1.0 + std::fabs(Rf[i])));
            const bool ok_f = err_f < 1e-5;
            std::printf("rect 127x37x53 %-26s max_rel_err=%.3g %s\n", "gemm_ikj(float)", err_f,
                        ok_f ? "OK" : "FAIL");
            failures += !ok_f;

#if defined(MATMUL_HAVE_NEON)
            std::fill(Cf.begin(), Cf.end(), 123.0f);
            gemm_neon_4x4(Cf.data(), Af.data(), Bf.data(), M, N, K);
            double err_n = 0.0;
            for (int i = 0; i < M * N; ++i)
                err_n = std::max(err_n, static_cast<double>(std::fabs(Cf[i] - Rf[i]))
                                            / (1.0 + std::fabs(Rf[i])));
            const bool ok_n = err_n < 1e-5;
            std::printf("rect 127x37x53 %-26s max_rel_err=%.3g %s\n", "gemm_neon_4x4(float)", err_n,
                        ok_n ? "OK" : "FAIL");
            failures += !ok_n;
#endif
        }

        return failures;
    }
} // namespace

int main(int argc, char **argv)
{
    if (argc > 1 && std::strcmp(argv[1], "--verify") == 0)
        return verify() == 0 ? 0 : 1;

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
