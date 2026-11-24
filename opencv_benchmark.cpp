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
    void gemm_rect_ikj_f32(const float *A, const float *B, float *C, int M, int N, int K)
    {
        gemm_ikj(C, A, B, M, N, K);
    }

#if defined(MATMUL_HAVE_NEON)
    void gemm_rect_neon_f32(const float *A, const float *B, float *C, int M, int N, int K)
    {
        gemm_neon_4x4(C, A, B, M, N, K);
    }
#endif

    // 2D Convolution / Correlation helpers on CV_32FC1

    void conv2d_naive_f32(const cv::Mat &src, const cv::Mat &kernel, cv::Mat &dst)
    {
        const int H = src.rows;
        const int W = src.cols;
        const int K = kernel.rows;
        const int pad = K / 2;

        dst.create(H, W, CV_32FC1);
        for (int y = 0; y < H; ++y)
        {
            float *dst_row = dst.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                float sum = 0.0f;
                for (int ky = 0; ky < K; ++ky)
                {
                    const int sy = y + ky - pad;
                    if (sy < 0 || sy >= H)
                        continue;
                    const float *src_row = src.ptr<float>(sy);
                    const float *ker_row = kernel.ptr<float>(ky);
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

    void im2col_f32(const cv::Mat &src, int K, std::vector<float> &col)
    {
        const int H = src.rows;
        const int W = src.cols;
        const int pad = K / 2;
        const int kk = K * K;

        col.resize(static_cast<std::size_t>(H) * W * kk);
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                float *dst_patch = col.data() + (static_cast<std::size_t>(y) * W + x) * kk;
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
                    const float *src_row = src.ptr<float>(sy);
                    for (int kx = 0; kx < K; ++kx)
                    {
                        const int sx = x + kx - pad;
                        dst_patch[idx++] = (sx >= 0 && sx < W) ? src_row[sx] : 0.0f;
                    }
                }
            }
        }
    }

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
            cv::gemm(A, B, 1.0, cv::Mat(), 0.0, C);
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

#if defined(MATMUL_HAVE_ACCELERATE)
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

    // Benchmarks: Multi-filter 2D Convolution (5x5, 32 filters)
    //   1. Naive 4-loop convolution
    //   2. OpenCV cv::filter2D (32 passes)
    //   3. im2col + gemm_rect_ikj_f32
    //   4. im2col + gemm_rect_neon_f32

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

    void bm_conv2d_naive(benchmark::State &state)
    {
        const int H = static_cast<int>(state.range(0));
        const int W = H;
        cv::Mat src = make_test_image_f32(H, W);
        cv::Mat weights = make_filter_bank_f32(kConvKernel, kNumFilters);

        cv::Mat kernel0(kConvKernel, kConvKernel, CV_32FC1);
        for (int i = 0; i < kConvKernel * kConvKernel; ++i)
            kernel0.at<float>(i / kConvKernel, i % kConvKernel) = weights.at<float>(i, 0);

        cv::Mat ref, dst;
        cv::filter2D(src, ref, CV_32F, kernel0, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);
        conv2d_naive_f32(src, kernel0, dst);
        if (cv::norm(dst, ref, cv::NORM_INF) > 1e-4)
        {
            state.SkipWithError("conv2d_naive differs from cv::filter2D");
            return;
        }

        for (auto _ : state)
        {
            for (int f = 0; f < kNumFilters; ++f)
            {
                for (int i = 0; i < kConvKernel * kConvKernel; ++i)
                    kernel0.at<float>(i / kConvKernel, i % kConvKernel) = weights.at<float>(i, f);
                conv2d_naive_f32(src, kernel0, dst);
            }
            benchmark::DoNotOptimize(dst.data);
        }
        const double flops = 2.0 * H * W * kConvKernel * kConvKernel * kNumFilters;
        state.counters["GFLOP/s"] =
            benchmark::Counter(flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    template <void (*GemmFn)(const float *, const float *, float *, int, int, int)>
    void bm_conv2d_im2col(benchmark::State &state)
    {
        const int H = static_cast<int>(state.range(0));
        const int W = H;
        const int M = H * W;
        const int K = kConvKernel * kConvKernel;
        const int N = kNumFilters;

        cv::Mat src = make_test_image_f32(H, W);
        cv::Mat weights = make_filter_bank_f32(kConvKernel, kNumFilters);

        // Verify filter 0 against cv::filter2D
        cv::Mat kernel0(kConvKernel, kConvKernel, CV_32FC1);
        for (int i = 0; i < K; ++i)
            kernel0.at<float>(i / kConvKernel, i % kConvKernel) = weights.at<float>(i, 0);
        cv::Mat ref;
        cv::filter2D(src, ref, CV_32F, kernel0, cv::Point(-1, -1), 0.0, cv::BORDER_CONSTANT);

        std::vector<float> col;
        std::vector<float> out(static_cast<std::size_t>(M) * N);
        im2col_f32(src, kConvKernel, col);
        GemmFn(col.data(), weights.ptr<float>(), out.data(), M, N, K);

        double max_err = 0.0;
        for (int y = 0; y < H; ++y)
        {
            const float *ref_row = ref.ptr<float>(y);
            for (int x = 0; x < W; ++x)
            {
                const float got = out[(static_cast<std::size_t>(y) * W + x) * N];
                max_err = std::max(max_err, static_cast<double>(std::fabs(got - ref_row[x])));
            }
        }
        if (max_err > 1e-4)
        {
            state.SkipWithError("im2col + GEMM differs from cv::filter2D");
            return;
        }

        for (auto _ : state)
        {
            im2col_f32(src, kConvKernel, col);
            GemmFn(col.data(), weights.ptr<float>(), out.data(), M, N, K);
            benchmark::DoNotOptimize(out.data());
        }
        const double flops = 2.0 * M * K * N;
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

    // Benchmarks: Pixel Color Matrix Transform ((H*W) x 3) * (3 x 3) vs cv::transform

    void color_transform_gemm_f32(const cv::Mat &src, const float *M3x3, cv::Mat &dst)
    {
        const int H = src.rows;
        const int W = src.cols;
        dst.create(H, W, CV_32FC3);

        for (int y = 0; y < H; ++y)
        {
            const float *s = src.ptr<float>(y);
            float *d = dst.ptr<float>(y);
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

    void bm_color_transform_cv(benchmark::State &state)
    {
        const int H = static_cast<int>(state.range(0));
        const int W = H;
        cv::Mat src(H, W, CV_32FC3), dst(H, W, CV_32FC3);
        cv::randu(src, 0.0f, 1.0f);

        const cv::Matx33f sepia(0.393f, 0.769f, 0.189f, 0.349f, 0.686f, 0.168f, 0.272f, 0.534f,
                                0.131f);

        for (auto _ : state)
        {
            cv::transform(src, dst, sepia);
            benchmark::DoNotOptimize(dst.data);
        }
        const double flops = 2.0 * H * W * 3 * 3;
        state.counters["GFLOP/s"] =
            benchmark::Counter(flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    void bm_color_transform_gemm(benchmark::State &state)
    {
        const int H = static_cast<int>(state.range(0));
        const int W = H;
        cv::Mat src(H, W, CV_32FC3), dst, ref;
        cv::randu(src, 0.0f, 1.0f);

        const float M[9] = {0.393f, 0.769f, 0.189f, 0.349f, 0.686f, 0.168f, 0.272f, 0.534f, 0.131f};
        const cv::Matx33f sepia(M);

        cv::transform(src, ref, sepia);
        color_transform_gemm_f32(src, M, dst);
        if (cv::norm(dst, ref, cv::NORM_INF) > 1e-4)
        {
            state.SkipWithError("color_transform_gemm_f32 differs from cv::transform");
            return;
        }

        for (auto _ : state)
        {
            color_transform_gemm_f32(src, M, dst);
            benchmark::DoNotOptimize(dst.data);
        }
        const double flops = 2.0 * H * W * 3 * 3;
        state.counters["GFLOP/s"] =
            benchmark::Counter(flops * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
    }

    BENCHMARK(bm_cv_gemm_f64)->Apply(configure_gemm);
#if defined(MATMUL_HAVE_NEON)
    BENCHMARK(bm_neon_4x4_cv_mat_f64)->Apply(configure_gemm);
#endif
#if defined(MATMUL_HAVE_ACCELERATE)
    BENCHMARK(bm_accelerate_cv_mat_f64)->Apply(configure_gemm);
#endif

    BENCHMARK(bm_conv2d_naive)->Apply(configure_conv);
    BENCHMARK(bm_conv2d_filter2d)->Apply(configure_conv);
    BENCHMARK_TEMPLATE(bm_conv2d_im2col, gemm_rect_ikj_f32)
        ->Name("bm_conv2d_im2col_ikj")
        ->Apply(configure_conv);
#if defined(__aarch64__)
    BENCHMARK_TEMPLATE(bm_conv2d_im2col, gemm_rect_neon_f32)
        ->Name("bm_conv2d_im2col_neon")
        ->Apply(configure_conv);
#endif

    BENCHMARK(bm_color_transform_cv)->Apply(configure_gemm);
    BENCHMARK(bm_color_transform_gemm)->Apply(configure_gemm);
} // namespace
BENCHMARK_MAIN();
