
#include "matrix-multiply.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <concepts>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace calculate_matrix;

namespace
{

    template <typename Fn>
    concept RowPtrKernel =
        std::invocable<Fn, double *const *, const double *const *, const double *const *, int>;

    template <typename Fn>
    concept FlatKernel = std::invocable<Fn, const double *, const double *, double *, int, int>;

    class Matrix
    {
      public:
        explicit Matrix(int n) : n_(n), rows_(new double *[n])
        {
            for (int i = 0; i < n; ++i)
                rows_[i] = new double[n]();
        }
        ~Matrix()
        {
            for (int i = 0; i < n_; ++i)
                delete[] rows_[i];
            delete[] rows_;
        }
        Matrix(const Matrix &) = delete;
        Matrix &operator=(const Matrix &) = delete;

        double **data()
        {
            return rows_;
        }

        void fill_random()
        {
            for (int i = 0; i < n_; ++i)
                for (int j = 0; j < n_; ++j)
                    rows_[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
        }

        void reset()
        {
            for (int i = 0; i < n_; ++i)
                for (int j = 0; j < n_; ++j)
                    rows_[i][j] = NAN;
        }

      private:
        int n_;
        double **rows_;
    };

    bool matrix_equal(Matrix &reference, Matrix &actual, int n)
    {
        const double absolute_tolerance = 1e-9;
        const double relative_tolerance = 1e-9;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                const double expected = reference.data()[i][j];
                const double observed = actual.data()[i][j];
                const double allowed =
                    absolute_tolerance + relative_tolerance * std::fabs(expected);
                if (!std::isfinite(expected) || !std::isfinite(observed)
                    || std::fabs(expected - observed) > allowed)
                    return false;
            }
        }
        return true;
    }

    struct Inputs
    {
        Matrix A, B, expected;

        explicit Inputs(int n) : A(n), B(n), expected(n)
        {
            std::srand(0);
            A.fill_random();
            B.fill_random();
            bad_matrix_multiply(expected.data(), A.data(), B.data(), n);
        }
    };

    Inputs &inputs_for(int n)
    {
        static std::map<int, std::unique_ptr<Inputs>> cache;
        auto &entry = cache[n];
        if (!entry)
            entry = std::make_unique<Inputs>(n);
        return *entry;
    }

    template <RowPtrKernel Fn> void multiply(benchmark::State &state, Fn fn)
    {
        const int n = static_cast<int>(state.range(0));
        Inputs &in = inputs_for(n);
        Matrix C(n);

        C.reset();
        fn(C.data(), in.A.data(), in.B.data(), n);
        if (!matrix_equal(in.expected, C, n))
        {
            state.SkipWithError("result differs from bad_matrix_multiply");
            return;
        }

        for (auto _ : state)
        {
            fn(C.data(), in.A.data(), in.B.data(), n);
            benchmark::DoNotOptimize(C.data()[0][0]);
        }

        if (!matrix_equal(in.expected, C, n))
            state.SkipWithError("result differs from bad_matrix_multiply");

        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * static_cast<double>(n) * n * n * 1e-9,
                               benchmark::Counter::kIsIterationInvariantRate);
    }

    void configure(benchmark::Benchmark *b)
    {
        b->Arg(256)
            ->Arg(512)
            ->Arg(500)
            ->Arg(1000)
            ->Arg(1024)
            ->Arg(2000)
            ->Arg(2048)
            ->Unit(benchmark::kMillisecond)
            ->Repetitions(5)
            ->ReportAggregatesOnly(true);
    }

    template <FlatKernel Fn> void multiply_flat(benchmark::State &state, Fn fn)
    {
        const int n = static_cast<int>(state.range(0));
        Inputs &in = inputs_for(n);
        const std::size_t nn = static_cast<std::size_t>(n) * n;
        std::vector<double> A(nn), B(nn), C(nn);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                A[i * n + j] = in.A.data()[i][j];
                B[i * n + j] = in.B.data()[i][j];
            }

        // Every function overwrites C, so stale values must not survive a call.
        std::fill(C.begin(), C.end(), NAN);

        for (auto _ : state)
        {
            fn(A.data(), B.data(), C.data(), n, BS);
            benchmark::DoNotOptimize(C[0]);
        }

        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
            {
                const double expected = in.expected.data()[i][j];
                if (std::fabs(expected - C[i * n + j]) > 1e-9 + 1e-9 * std::fabs(expected))
                {
                    state.SkipWithError("result differs from bad_matrix_multiply");
                    return;
                }
            }

        state.SetItemsProcessed(state.iterations() * int64_t(n) * n * n);
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * static_cast<double>(n) * n * n * 1e-9,
                               benchmark::Counter::kIsIterationInvariantRate);
    }

    void schur_as_matmul(benchmark::State &state)
    {
        const int N = static_cast<int>(state.range(0));
        const int M = 2 * N;

        const std::size_t nn = static_cast<std::size_t>(N) * N;
        const std::size_t mm = static_cast<std::size_t>(M) * M;

        std::vector<double> A(nn), B(nn);
        std::srand(0);
        for (double &x : A)
            x = static_cast<double>(std::rand()) / RAND_MAX;
        for (double &x : B)
            x = static_cast<double>(std::rand()) / RAND_MAX;

        // C = A * B using simple triple loop
        std::vector<double> C(nn, 0.0);
        for (int i = 0; i < N; ++i)
            for (int k = 0; k < N; ++k)
                for (int j = 0; j < N; ++j)
                    C[i * N + j] += A[i * N + k] * B[k * N + j];

        // Packing: B → top-right block (U12), A → bottom-left (L21)
        std::vector<double> big(mm, 0.0);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                big[i * M + (N + j)] = B[i * N + j];
                big[(N + i) * M + j] = A[i * N + j];
            }

        auto zero_tail = [&]
        {
            for (int i = N; i < M; ++i)
                std::fill_n(&big[i * M + N], N, 0.0);
        };

        // Check: tail == -C
        gemm_sub_tiled(big.data(), M, 0, N, BS);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                const double got = -big[(N + i) * M + (N + j)];
                const double exp = C[i * N + j];
                if (std::fabs(got - exp) > 1e-9 + 1e-9 * std::fabs(exp))
                {
                    state.SkipWithError("tail != -(A * B)");
                    return;
                }
            }

        for (auto _ : state)
        {
            zero_tail(); // same zero_matrix(C, N)
            gemm_sub_tiled(big.data(), M, 0, N, BS);
            benchmark::DoNotOptimize(big.data());
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * int64_t(N) * N * N);
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * static_cast<double>(N) * N * N * 1e-9,
                               benchmark::Counter::kIsIterationInvariantRate);
    }

    const std::string bs = std::to_string(BS);
    const std::string blocked_name = "blocked_bs" + bs;
    const std::string tiled_name = "tiled_bs" + bs;
    const std::string tiled_flat_name = "tiled_flat_bs" + bs;
    const std::string gemm_name = "gemm_sub_tiled_bs" + bs;

    [[maybe_unused]] const auto *naive =
        benchmark::RegisterBenchmark("naive_ijk",
                                     [](benchmark::State &s) { multiply(s, bad_matrix_multiply); })
            ->Apply(configure);
    [[maybe_unused]] const auto *fast =
        benchmark::RegisterBenchmark("fast_ikj",
                                     [](benchmark::State &s) { multiply(s, fast_matrix_multiply); })
            ->Apply(configure);
    [[maybe_unused]] const auto *tiled =
        benchmark::RegisterBenchmark(tiled_name,
                                     [](benchmark::State &s) { multiply(s, matmul_tiled); })
            ->Apply(configure);
    [[maybe_unused]] const auto *blocked =
        benchmark::RegisterBenchmark(
            blocked_name,
            [](benchmark::State &s)
            {
                multiply_flat(s, [](const double *A, const double *B, double *C, int n, int)
                              { blocked_matrix_multiply(C, A, B, n); });
            })
            ->Apply(configure);
    [[maybe_unused]] const auto *tiled_flat =
        benchmark::RegisterBenchmark(tiled_flat_name,
                                     [](benchmark::State &s)
                                     { multiply_flat(s, matmul_tiled_flat); })
            ->Apply(configure);
    [[maybe_unused]] const auto *gemm =
        benchmark::RegisterBenchmark(gemm_name, schur_as_matmul)->Apply(configure);

    const std::string blocked_k4_name = "blocked_k4_bs" + bs;
    [[maybe_unused]] const auto *blocked_k4 =
        benchmark::RegisterBenchmark(
            blocked_k4_name,
            [](benchmark::State &s)
            {
                multiply_flat(s, [](const double *A, const double *B, double *C, int n, int)
                              { blocked_k4_matrix_multiply(C, A, B, n); });
            })
            ->Apply(configure);

    const std::string blocked_k8_name = "blocked_k8_bs" + bs;
    [[maybe_unused]] const auto *blocked_k8 =
        benchmark::RegisterBenchmark(
            blocked_k8_name,
            [](benchmark::State &s)
            {
                multiply_flat(s, [](const double *A, const double *B, double *C, int n, int tile)
                              { blocked_matrix_multiply_k8(C, A, B, n, tile); });
            })
            ->Apply(configure);

    const std::string tiled_flat_v2_name = "tiled_flat_v2_bs" + bs;
    [[maybe_unused]] const auto *tiled_flat_v2 =
        benchmark::RegisterBenchmark(tiled_flat_v2_name,
                                     [](benchmark::State &s)
                                     { multiply_flat(s, matmul_tiled_flat_v2); })
            ->Apply(configure);

    const std::string blocked_k4_ikj_name = "blocked_k4_ikj_bs" + bs;
    [[maybe_unused]] const auto *blocked_k4_ikj =
        benchmark::RegisterBenchmark(
            blocked_k4_ikj_name,
            [](benchmark::State &s)
            {
                multiply_flat(s, [](const double *A, const double *B, double *C, int n, int)
                              { blocked_k4_ikj_matrix_multiply(C, A, B, n); });
            })
            ->Apply(configure);

    [[maybe_unused]] const auto *ikj_k4 =
        benchmark::RegisterBenchmark(
            "ikj_k4",
            [](benchmark::State &s)
            {
                multiply_flat(s, [](const double *A, const double *B, double *C, int n, int)
                              { ikj_k4_matrix_multiply(C, A, B, n); });
            })
            ->Apply(configure);

#if defined(MATMUL_HAVE_NEON)
    const std::string neon_4x4_name = "neon_4x4_bs" + bs;
    [[maybe_unused]] const auto *neon_4x4 =
        benchmark::RegisterBenchmark(
            neon_4x4_name,
            [](benchmark::State &s)
            {
                multiply_flat(s, [](const double *A, const double *B, double *C, int n, int)
                              { neon_4x4_matrix_multiply(C, A, B, n); });
            })
            ->Apply(configure);
#endif

#if defined(MATMUL_HAVE_ACCELERATE)
    [[maybe_unused]] const auto *accelerate_dgemm =
        benchmark::RegisterBenchmark(
            "accelerate_dgemm",
            [](benchmark::State &s)
            {
                multiply_flat(s, [](const double *A, const double *B, double *C, int n, int)
                              { accelerate_dgemm_matrix_multiply(C, A, B, n); });
            })
            ->Apply(configure);
#endif
}
