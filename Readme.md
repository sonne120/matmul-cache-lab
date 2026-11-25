## Fast Matrix Multiplication in C++20, Compared with OpenCV

A C++20 study of dense matrix multiplication (`C = A × B`) built up step by step in [`src/matrix-multiply.cpp`](./src/matrix-multiply.cpp), verified with `--verify`, and benchmarked against Apple Accelerate / CBLAS ([`src/accelerate-gemm.cpp`](./src/accelerate-gemm.cpp)) and OpenCV 4 ([`src/opencv-gemm.cpp`](./src/opencv-gemm.cpp)).

## 1. Matrix Multiplication Progression

Everything in this section is custom code: steps 1–6 are written by hand in [`src/matrix-multiply.cpp`](./src/matrix-multiply.cpp) and call no library. OpenCV is not used here at all; it appears only in [section 2](#2-opencv-comparison). The one library result in the table below is `accelerate_dgemm`, which is Apple Accelerate, not OpenCV.

| Code | Where | Used in |
| :--- | :--- | :--- |
| **Custom** (hand-written C++ and NEON) | [`src/matrix-multiply.cpp`](./src/matrix-multiply.cpp) | Sections 1 and 2 |
| **Apple Accelerate** (`cblas_dgemm`) | [`src/accelerate-gemm.cpp`](./src/accelerate-gemm.cpp) | Sections 1 and 2, as the vendor baseline |
| **OpenCV 4** (`cv::gemm`, `cv::filter2D`, `cv::transform`, …) | [`src/opencv-gemm.cpp`](./src/opencv-gemm.cpp) | Section 2 only (`matmul-opencv` target) |

1. **Naive (`bad_matrix_multiply`):** `i-j-k` dot products on `double**`; `B` is read with column stride. *Custom.*
2. **Loop interchange (`fast_matrix_multiply`):** `i-k-j` order; `B` and `C` rows stream sequentially (**7.7×** faster at `N = 1024`). *Custom.*
3. **Cache blocking (`matmul_tiled*`, `blocked_matrix_multiply`):** `BS × BS` tiling (`BS = 16..512`) and flat `double*` storage (`i * N + j`). *Custom.*
4. **`k`-unrolling (`ikj_k4_*`, `blocked_k4_*`, `blocked_matrix_multiply_k8`):** advances `k` by 4 (or 8) to cut `C` load/store traffic, with scalar cleanup. *Custom.*
5. **NEON 4×4 micro-kernel (`neon_4x4_matrix_multiply`):** `jj-kk` tiling, contiguous 4-column `B` packing, and 8 `float64x2_t` accumulators (`vfmaq_n_f64`), plus scalar edge cleanup (**16.5×** faster). *Custom.*
6. **Rectangular & vision helpers:** `gemm_ikj` (`double`/`float`), `gemm_neon_4x4` (`float`), `gemm_sub_tiled` (LU update), `conv2d_naive_f32`, `im2col_f32`, and `color_transform_gemm_f32`. *Custom; compared in section 2 with OpenCV's `cv::gemm`, `cv::filter2D`, and `cv::transform`.*

| Variant | Code | Technique | N = 1024 Time | N = 1024 GFLOP/s | N = 2048 Time | N = 2048 GFLOP/s | Speedup vs `naive_ijk` |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `naive_ijk` | Custom | `i-j-k`, `double**` | 1218 ms | 1.76 | 28,399 ms* | 0.60* | 1.0× |
| `blocked_bs64` | Custom | `si-sk-sj`, flat `double*` | 222 ms | 9.67 | — | — | 5.5× |
| `fast_ikj` | Custom | `i-k-j`, `double**` | 158 ms | 13.6 | 1316 ms | 13.1 | 7.7× |
| `ikj_k4` | Custom | `i-k-j` + `k×4` unroll | 127 ms | 16.9 | 1377 ms | 12.5 | 9.6× |
| `neon_4x4_bs64` | Custom | 4×4 NEON FMA + packed `B` | **73.9 ms** | **29.1** | **702 ms** | **25.1** | **16.5×** |
| `accelerate_dgemm` | Apple Accelerate | `cblas_dgemm` (`VECLIB_THREADS=1`) | **6.35 ms** | **339.5** | **56.1 ms** | **307.1** | **191.8×** |

*\*`naive_ijk` at N = 2048 was measured in an earlier `-O2` run.*

---

## 2. OpenCV Comparison

[`src/opencv_benchmark.cpp`](./src/opencv_benchmark.cpp) benchmarks OpenCV 4 (`cv::gemm`, `cv::filter2D`, `cv::transform`, `cv::MatExpr`, `cv::mulTransposed`, `cv::Matx`) alongside the standalone C++/NEON benchmarks in [`src/google_benchmark.cpp`](./src/google_benchmark.cpp):

| Workload | Variant | Size / Type | Time | GFLOP/s | Notes |
| :--- | :--- | :--- | ---: | ---: | :--- |
| Dense GEMM | `accelerate_cv_mat_f64` | `1024×1024`, `CV_64F` | **3.33 ms** | **662.0** | Apple Accelerate (~3.2× faster than `cv::gemm`) |
| Dense GEMM | `cv::gemm` | `1024×1024`, `CV_64F` | 10.7 ms | 408.3 | OpenCV baseline (~7.1× faster than NEON) |
| Dense GEMM | `neon_4x4_cv_mat_f64` | `1024×1024`, `CV_64F` | 76.3 ms | 28.2 | Hand-written NEON 4×4 on `cv::Mat` |
| 2D Conv (32× `5×5`) | `cv::filter2D` | `512×512`, `CV_32F` | **10.4 ms** | **40.5** | OpenCV baseline |
| 2D Conv (32× `5×5`) | `conv2d_im2col_neon` | `512×512`, `CV_32F` | 13.6 ms | 30.8 | `im2col` + NEON `f32` (~11.8× faster than naive) |
| 2D Conv (32× `5×5`) | `conv2d_im2col_ikj` | `512×512`, `CV_32F` | 31.9 ms | 13.2 | `im2col` + scalar `i-k-j` `f32` |
| 2D Conv (32× `5×5`) | `bm_conv2d_naive` | `512×512`, `CV_32F` | 160 ms | 2.63 | 4-loop scalar baseline |
| 3×3 Color Transform | `bm_color_transform_gemm` | `1024×1024`, `CV_32FC3` | **0.325 ms** | **58.1** | **4.0×** faster than `cv::transform` |
| 3×3 Color Transform | `cv::transform` | `1024×1024`, `CV_32FC3` | 1.30 ms | 14.5 | OpenCV baseline |
| Rectangular GEMM | `opencv_gemm_f32` | `1024×512 × 512×768`, `CV_32F` | 1.68 ms | 884.3 | `cv::gemm` single-precision |
| Rectangular GEMM | `opencv_matmul_expr_f64` | `1024×512 × 512×768`, `CV_64F` | 3.42 ms | 428.4 | `cv::MatExpr` (`A * B`) |
| Rectangular GEMM | `opencv_gemm_axpb_f64` | `1024×512 × 512×768`, `CV_64F` | 3.63 ms | 360.6 | `1.5 * A * B^T + 0.5 * D` (`GEMM_2_T`) |
| Complex GEMM | `opencv_gemm_complex_f64` | `1024×512 × 512×768`, `CV_64FC2` | 14.1 ms | 433.9 | 2-channel complex `cv::gemm` |
| Gram Matrix | `opencv_mul_transposed_f64` | `2048×1024`, `CV_64F` | 20.4 ms | 277.2 | `A^T * A` via `cv::mulTransposed` |
| Fixed-Size `Matx` | `opencv_matx_mul_2x3_3x4` | `2×3 × 3×4`, `float` | 0.898 ns | — | Stack-allocated `cv::Matx` |

---

## Build, Verify & Run

```sh
cmake -S . -B build
cmake --build build -j 8

# Verify correctness across all variants and block sizes
cmake --build build --target verify-all

# Run benchmarks
cmake --build build --target run-bs64
cmake --build build --target run-opencv

# Build and run with Clang
cmake -S . -B build-clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build-clang -j 8
./build-clang/matmul-bs64 --verify
./build-clang/matmul-bs64

# Build and run with GCC 
cmake -S . -B build-gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build-gcc -j 8
./build-gcc/matmul-bs64 --verify
./build-gcc/matmul-bs64

# Build and run on Windows
cmake -S . -B build-win
cmake --build build-win --config Release --target matmul-bs64 -j 8
.\build-win\Release\matmul-bs64.exe --verify
.\build-win\Release\matmul-bs64.exe
```
