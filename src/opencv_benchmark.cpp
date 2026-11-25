#include "matrix-multiply.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <cstdlib>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

using namespace calculate_matrix;

namespace
{
    // 2D Convolution / Correlation helpers on CV_32FC1

    cv::Mat make_test_image_f32(int H, int W)
    {
        cv::Mat img(H, W, CV_32FC1);
        std::srand(42);
        for (int y = 0; y < H; ++y)
        {
            float *row = img.ptr<float>(y);
            for (int x = 0; x < W; ++x)
                row[x] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        }
        return img;
    }

    cv::Mat make_filter_bank_f32(int K, int num_filters)
    {
        // Shape: (K*K) x num_filters, each column is a normalized filter
        cv::Mat weights(K * K, num_filters, CV_32FC1);
        std::srand(123);
        for (int f = 0; f < num_filters; ++f)
        {
            float sum = 0.0f;
            for (int i = 0; i < K * K; ++i)
            {
                const float v = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
                weights.at<float>(i, f) = v;
                sum += v;
            }
            for (int i = 0; i < K * K; ++i)
                weights.at<float>(i, f) /= sum;
        }
        return weights;
    }

    // Benchmarks: Dense CV_64F cv::Mat GEMM vs cv::gemm

    void bm_cv_gemm_f64(benchmark::State &state)
    {
        const int n = static_cast<int>(state.range(0));
        cv::Mat A(n, n, CV_64F), B(n, n, CV_64F), C(n, n, CV_64F);
        cv::randu(A, 0.0, 1.0);
        cv::randu(B, 0.0, 1.0);

        for (auto _ : state)
        {
            opencv_gemm_f64(A, B, C);
            benchmark::DoNotOptimize(C.data);
        }
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * static_cast<double>(n) * n * n * 1e-9,
                               benchmark::Counter::kIsIterationInvariantRate);
    }

#if defined(MATMUL_HAVE_NEON)
    void bm_neon_4x4_cv_mat_f64(benchmark::State &state)
    {
        const int n = static_cast<int>(state.range(0));
        cv::Mat A(n, n, CV_64F), B(n, n, CV_64F), C(n, n, CV_64F), ref;
        cv::randu(A, 0.0, 1.0);
        cv::randu(B, 0.0, 1.0);
        cv::gemm(A, B, 1.0, cv::Mat(), 0.0, ref);

        neon_4x4_matrix_multiply(C.ptr<double>(), A.ptr<double>(), B.ptr<double>(), n);
        const double max_err = cv::norm(C, ref, cv::NORM_INF);
        if (max_err > 1e-9)
        {
            state.SkipWithError("neon_4x4 differs from cv::gemm");
            return;
        }

        for (auto _ : state)
        {
            neon_4x4_matrix_multiply(C.ptr<double>(), A.ptr<double>(), B.ptr<double>(), n);
            benchmark::DoNotOptimize(C.data);
        }
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * static_cast<double>(n) * n * n * 1e-9,
                               benchmark::Counter::kIsIterationInvariantRate);
    }
#endif

#if defined(MATMUL_HAVE_ACCELERATE) || defined(MATMUL_HAVE_CBLAS)
    void bm_accelerate_cv_mat_f64(benchmark::State &state)
    {
        const int n = static_cast<int>(state.range(0));
        cv::Mat A(n, n, CV_64F), B(n, n, CV_64F), C(n, n, CV_64F), ref;
        cv::randu(A, 0.0, 1.0);
        cv::randu(B, 0.0, 1.0);
        cv::gemm(A, B, 1.0, cv::Mat(), 0.0, ref);

        accelerate_dgemm_matrix_multiply(C.ptr<double>(), A.ptr<double>(), B.ptr<double>(), n);
        if (cv::norm(C, ref, cv::NORM_INF) > 1e-9)
        {
            state.SkipWithError("accelerate_dgemm differs from cv::gemm");
            return;
        }

        for (auto _ : state)
        {
            accelerate_dgemm_matrix_multiply(C.ptr<double>(), A.ptr<double>(), B.ptr<double>(), n);
            benchmark::DoNotOptimize(C.data);
        }
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * static_cast<double>(n) * n * n * 1e-9,
                               benchmark::Counter::kIsIterationInvariantRate);
    }
#endif

    // Benchmarks: Multi-filter 2D Convolution (5x5, 32 filters) via OpenCV cv::filter2D

    constexpr int kConvKernel = 5;
    constexpr int kNumFilters = 32;

    void bm_conv2d_filter2d(benchmark::State &state)
    {
        const int H = static_cast<int>(state.range(0));
        const int W = H;
        cv::Mat src = make_test_image_f32(H, W);
        cv::Mat weights = make_filter_bank_f32(kConvKernel, kNumFilters);

        std::vector<cv::Mat> kernels(kNumFilters);
        for (int f = 0; f < kNumFilters; ++f)
        {
            kernels[f].create(kConvKernel, kConvKernel, CV_32FC1);
            for (int i = 0; i < kConvKernel * kConvKernel; ++i)
                kernels[f].at<float>(i / kConvKernel, i % kConvKernel) = weights.at<float>(i, f);
        }

        cv::Mat dst(H, W, CV_32FC1);
        for (auto _ : state)
        {
            for (int f = 0; f < kNumFilters; ++f)
                cv::filter2D(src, dst, CV_32F, kernels[f], cv::Point(-1, -1), 0.0,
                             cv::BORDER_CONSTANT);
            benchmark::DoNotOptimize(dst.data);
        }
        const double flops = 2.0 * H * W * kConvKernel * kConvKernel * kNumFilters;
        state.counters["GFLOP/s"] =
            benchmark::Counter(flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    void configure_gemm(benchmark::Benchmark *b)
    {
        b->Arg(256)
            ->Arg(512)
            ->Arg(1024)
            ->Unit(benchmark::kMillisecond)
            ->Repetitions(5)
            ->ReportAggregatesOnly(true);
    }

    void configure_conv(benchmark::Benchmark *b)
    {
        b->Arg(256)
            ->Arg(512)
            ->Unit(benchmark::kMillisecond)
            ->Repetitions(5)
            ->ReportAggregatesOnly(true);
    }

    // Pixel Color Matrix Transform via OpenCV cv::transform

    void bm_color_transform_cv(benchmark::State &state)
    {
        const int H = static_cast<int>(state.range(0));
        const int W = H;
        cv::Mat src(H, W, CV_32FC3), dst(H, W, CV_32FC3);
        cv::randu(src, 0.0f, 1.0f);

        const cv::Matx33f sepia_matx(0.393f, 0.769f, 0.189f, 0.349f, 0.686f, 0.168f, 0.272f,
                                     0.534f, 0.131f);
        const cv::Mat sepia(sepia_matx);

        for (auto _ : state)
        {
            opencv_transform_f32(src, sepia, dst);
            benchmark::DoNotOptimize(dst.data);
        }
        const double flops = 2.0 * H * W * 3 * 3;
        state.counters["GFLOP/s"] =
            benchmark::Counter(flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    //  Rectangular & specialized OpenCV matrix multiplication variants 

    void bm_cv_gemm_f32_rect(benchmark::State &state)
    {
        const int M = static_cast<int>(state.range(0));
        const int K = M / 2;
        const int N = (M * 3) / 4;
        cv::Mat A(M, K, CV_32F), B(K, N, CV_32F), C(M, N, CV_32F);
        cv::randu(A, 0.0f, 1.0f);
        cv::randu(B, 0.0f, 1.0f);

        for (auto _ : state)
        {
            opencv_gemm_f32(A, B, C);
            benchmark::DoNotOptimize(C.data);
        }
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * M * K * N * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    void bm_cv_gemm_axpb_f64_rect(benchmark::State &state)
    {
        const int M = static_cast<int>(state.range(0));
        const int K = M / 2;
        const int N = (M * 3) / 4;
        cv::Mat A(M, K, CV_64F), B(N, K, CV_64F), D(M, N, CV_64F), C(M, N, CV_64F);
        cv::randu(A, 0.0, 1.0);
        cv::randu(B, 0.0, 1.0);
        cv::randu(D, 0.0, 1.0);

        const cv::Mat ref = 1.5 * (A * B.t()) + 0.5 * D;
        opencv_gemm_axpb_f64(A, B, D, C, 1.5, 0.5, cv::GEMM_2_T);
        if (cv::norm(C, ref, cv::NORM_INF) > 1e-9)
        {
            state.SkipWithError("opencv_gemm_axpb_f64 differs from reference");
            return;
        }

        for (auto _ : state)
        {
            opencv_gemm_axpb_f64(A, B, D, C, 1.5, 0.5, cv::GEMM_2_T);
            benchmark::DoNotOptimize(C.data);
        }
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * M * K * N * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    void bm_cv_gemm_complex_f64_rect(benchmark::State &state)
    {
        const int M = static_cast<int>(state.range(0));
        const int K = M / 2;
        const int N = (M * 3) / 4;
        cv::Mat A(M, K, CV_64FC2), B(K, N, CV_64FC2), C(M, N, CV_64FC2);
        cv::randu(A, cv::Scalar::all(0.0), cv::Scalar::all(1.0));
        cv::randu(B, cv::Scalar::all(0.0), cv::Scalar::all(1.0));

        for (auto _ : state)
        {
            opencv_gemm_complex_f64(A, B, C);
            benchmark::DoNotOptimize(C.data);
        }
        // Complex multiply-add: 8 FLOPs per complex MAC
        state.counters["GFLOP/s"] =
            benchmark::Counter(8.0 * M * K * N * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    void bm_cv_matmul_expr_f64_rect(benchmark::State &state)
    {
        const int M = static_cast<int>(state.range(0));
        const int K = M / 2;
        const int N = (M * 3) / 4;
        cv::Mat A(M, K, CV_64F), B(K, N, CV_64F), C, ref;
        cv::randu(A, 0.0, 1.0);
        cv::randu(B, 0.0, 1.0);

        cv::gemm(A, B, 1.0, cv::Mat(), 0.0, ref);
        opencv_matmul_expr_f64(A, B, C);
        if (cv::norm(C, ref, cv::NORM_INF) > 1e-9)
        {
            state.SkipWithError("opencv_matmul_expr_f64 differs from cv::gemm");
            return;
        }

        for (auto _ : state)
        {
            opencv_matmul_expr_f64(A, B, C);
            benchmark::DoNotOptimize(C.data);
        }
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * M * K * N * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    void bm_cv_matmul_expr_f32_rect(benchmark::State &state)
    {
        const int M = static_cast<int>(state.range(0));
        const int K = M / 2;
        const int N = (M * 3) / 4;
        cv::Mat A(M, K, CV_32F), B(K, N, CV_32F), C, ref;
        cv::randu(A, 0.0f, 1.0f);
        cv::randu(B, 0.0f, 1.0f);

        cv::gemm(A, B, 1.0, cv::Mat(), 0.0, ref);
        opencv_matmul_expr_f32(A, B, C);
        if (cv::norm(C, ref, cv::NORM_INF) > 1e-3)
        {
            state.SkipWithError("opencv_matmul_expr_f32 differs from cv::gemm");
            return;
        }

        for (auto _ : state)
        {
            opencv_matmul_expr_f32(A, B, C);
            benchmark::DoNotOptimize(C.data);
        }
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * M * K * N * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    void bm_cv_mul_transposed_f64(benchmark::State &state)
    {
        const int N = static_cast<int>(state.range(0));
        const int M = N * 2;
        cv::Mat A(M, N, CV_64F), C(N, N, CV_64F), ref;
        cv::randu(A, 0.0, 1.0);

        cv::gemm(A, A, 1.0, cv::Mat(), 0.0, ref, cv::GEMM_1_T);
        opencv_mul_transposed_f64(A, C, true);
        if (cv::norm(C, ref, cv::NORM_INF) > 1e-8)
        {
            state.SkipWithError("opencv_mul_transposed_f64 differs from A^T * A");
            return;
        }

        for (auto _ : state)
        {
            opencv_mul_transposed_f64(A, C, true);
            benchmark::DoNotOptimize(C.data);
        }
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * M * N * N * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    void bm_cv_mul_transposed_delta_f64(benchmark::State &state)
    {
        const int N = static_cast<int>(state.range(0));
        const int M = N * 2;
        cv::Mat A(M, N, CV_64F), delta(1, N, CV_64F), C(N, N, CV_64F), ref;
        cv::randu(A, 0.0, 1.0);
        cv::randu(delta, 0.0, 1.0);

        const cv::Mat centered = A - cv::repeat(delta, M, 1);
        cv::gemm(centered, centered, 0.5, cv::Mat(), 0.0, ref, cv::GEMM_1_T);
        opencv_mul_transposed_f64(A, C, true, delta, 0.5);
        if (cv::norm(C, ref, cv::NORM_INF) > 1e-8)
        {
            state.SkipWithError("opencv_mul_transposed_f64(delta) differs from reference");
            return;
        }

        for (auto _ : state)
        {
            opencv_mul_transposed_f64(A, C, true, delta, 0.5);
            benchmark::DoNotOptimize(C.data);
        }
        state.counters["GFLOP/s"] =
            benchmark::Counter(2.0 * M * N * N * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    void bm_cv_matx_mul_2x3_3x4(benchmark::State &state)
    {
        const cv::Matx<float, 2, 3> A(1, 2, 3, 4, 5, 6);
        const cv::Matx<float, 3, 4> B(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);
        const cv::Matx<float, 2, 4> C = opencv_matx_mul_2x3_3x4(A, B);
        if (std::fabs(C(0, 0) - 74.0f) > 1e-5f || std::fabs(C(1, 3) - 218.0f) > 1e-5f)
        {
            state.SkipWithError("opencv_matx_mul_2x3_3x4 produced unexpected result");
            return;
        }

        for (auto _ : state)
        {
            auto res = opencv_matx_mul_2x3_3x4(A, B);
            benchmark::DoNotOptimize(res);
        }
    }

    BENCHMARK(bm_cv_gemm_f64)->Apply(configure_gemm);
#if defined(MATMUL_HAVE_NEON)
    BENCHMARK(bm_neon_4x4_cv_mat_f64)->Apply(configure_gemm);
#endif
#if defined(MATMUL_HAVE_ACCELERATE) || defined(MATMUL_HAVE_CBLAS)
    BENCHMARK(bm_accelerate_cv_mat_f64)->Apply(configure_gemm);
#endif

    BENCHMARK(bm_conv2d_filter2d)->Apply(configure_conv);
    BENCHMARK(bm_color_transform_cv)->Apply(configure_gemm);

    BENCHMARK(bm_cv_gemm_f32_rect)->Apply(configure_gemm);
    BENCHMARK(bm_cv_gemm_axpb_f64_rect)->Apply(configure_gemm);
    BENCHMARK(bm_cv_gemm_complex_f64_rect)->Apply(configure_gemm);
    BENCHMARK(bm_cv_matmul_expr_f64_rect)->Apply(configure_gemm);
    BENCHMARK(bm_cv_matmul_expr_f32_rect)->Apply(configure_gemm);
    BENCHMARK(bm_cv_mul_transposed_f64)->Apply(configure_gemm);
    BENCHMARK(bm_cv_mul_transposed_delta_f64)->Apply(configure_gemm);
    BENCHMARK(bm_cv_matx_mul_2x3_3x4)->Repetitions(5)->ReportAggregatesOnly(true);
}
BENCHMARK_MAIN();
