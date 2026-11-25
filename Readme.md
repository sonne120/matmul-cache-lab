## Fast Matrix Multiplication in C++20, Compared with OpenCV

A C++20 study of dense matrix multiplication (`C = A × B`). A hand-written multiply is built up step by step in [`src/matrix-multiply.cpp`](./src/matrix-multiply.cpp), from a naive triple loop to an ARM64 NEON kernel, and every step is measured against Apple Accelerate (`cblas_dgemm`) and OpenCV 4 (`cv::Mat`). 

### How the custom multiply is implemented

Each step is a separate single-threaded function, so every stage can be benchmarked on its own.

1. **Naive baseline (`bad_matrix_multiply`).** Loops in `i-j-k` order and computes each `C[i][j]` as a dot product. `A` is read along a row, but `B` is read down a column, so every step of the inner loop lands on a different row of `B`.
2. **Loop interchange (`fast_matrix_multiply`).** The same arithmetic in `i-k-j` order. `A[i][k]` is held in a local variable, and the inner loop walks one row of `B` and one row of `C` from left to right.
3. **Cache blocking (`matmul_tiled`, `matmul_tiled_flat`, `blocked_matrix_multiply`).** Three outer loops step over `BS × BS` tiles, and the `i-k-j` loops run inside one tile at a time. The variants differ in the order of the outer loops and in storage: the matrices move from an array of row pointers (`double**`) to one flat row-major array (`double*`) indexed as `i * N + j`. The tile size is the compile-time constant `BS`, and CMake builds one executable per value (16 to 512).
4. **`k`-loop unrolling (`ikj_k4_matrix_multiply`, `blocked_k4_*`, `blocked_matrix_multiply_k8`).** The `k` loop advances four (or eight) steps at a time: four (or eight) values from a row of `A` and four (or eight) rows of `B` are combined in a single inner pass, so each `C[i][j]` is loaded and stored once per four (or eight) multiply-adds. A scalar loop finishes the remaining 0–3 (or 0–7) values of `k`.
5. **NEON 4×4 micro-kernel (`neon_4x4_matrix_multiply`).** Works tile by tile in `jj-kk` order.
   - **Packing:** the current tile of `B` is copied into a contiguous buffer, one 4-column strip at a time, so the kernel reads it with unit stride regardless of `N`.
   - **Micro-kernel:** a 4×4 block of `C` is held in eight NEON registers (two `double`s each). For every `k`, one packed row of `B` is folded into the block with eight fused multiply-adds (`vfmaq_n_f64`), and the block is stored back once at the end.
   - **Edges:** rows and columns left over when a size is not a multiple of 4 are handled by scalar clean-up loops.

The same file also holds the rectangular and vision helper kernels (`gemm_ikj` for `double` and `float`, `gemm_neon_4x4` for `float`, `conv2d_naive_f32`, `im2col_f32`, `color_transform_gemm_f32`) and `gemm_sub_tiled`, an in-place LU trailing update (`A22 -= L21 × U12`).

Every kernel is verified against the reference implementation (`--verify`).

The custom kernels are also compared with the OpenCV library (`cv::Mat` data) in the optional `matmul-opencv` target ([`src/opencv_benchmark.cpp`](./src/opencv_benchmark.cpp), using wrappers in [`src/opencv-gemm.cpp`](./src/opencv-gemm.cpp)) alongside the standalone C++/NEON convolution and color-transform benchmarks in [`src/google_benchmark.cpp`](./src/google_benchmark.cpp):

- **Dense matrix product:** the NEON kernel against `cv::gemm`, which is ~7.1× faster at `N = 1024` (and `cblas_dgemm` is ~3.2× faster than `cv::gemm`).
- **2D convolution (32 filters, 5×5):** `im2col` followed by the NEON `f32` kernel (`13.6 ms`) against `cv::filter2D` (`10.4 ms`) at `512×512`.
- **Per-pixel color transform (3×3):** a small matrix product against `cv::transform`, where the hand-written version is **4.0× faster** (`0.325 ms` vs `1.30 ms` at `1024×1024`).
- **Additional OpenCV matrix variants:** rectangular `CV_32F`/`CV_64F` `cv::gemm` (including `alpha`/`beta` scaling and `GEMM_2_T`), `CV_64FC2` complex GEMM, `cv::MatExpr` (`A * B`), Gram matrix (`cv::mulTransposed`), and stack-allocated `cv::Matx`.



## 1. Matrix Multiplication Progression

Key optimizations implemented in [`src/matrix-multiply.cpp`](./src/matrix-multiply.cpp):

- **Loop interchange (`i-j-k` → `i-k-j`):** turns column-strided reads of `B` into sequential row streams (**7.7×** faster at `N = 1024`).
- **Flat row-major storage (`double*`) & cache blocking:** removes pointer-of-pointers indirection and tiles loops (`BS = 64`) to fit working sets in cache.
- **`k`-loop unrolling (`k × 4`):** reduces `C` load/store traffic by 4×.
- **ARM64 NEON 4×4 micro-kernel (`neon_4x4`):** packs `B` tiles contiguously and accumulates a 4×4 register tile using NEON FMA instructions (**16.5×–18.8×** over naive).
- **Apple Accelerate / CBLAS (`cblas_dgemm`):** vendor BLAS baseline in [`src/accelerate-gemm.cpp`](./src/accelerate-gemm.cpp) (`VECLIB_MAXIMUM_THREADS=1` on macOS, or OpenBLAS/CBLAS on Linux/Windows).

| Variant            | Technique                             |        N = 1024 Time | N = 1024 GFLOP/s |        N = 2048 Time | N = 2048 GFLOP/s | Speedup vs `naive_ijk` (N = 1024) |
| :----------------- | :------------------------------------ | -------------------: | ---------------: | -------------------: | ---------------: | --------------------------------: |
| `naive_ijk`        | `i-j-k`, `double**`                   |              1218 ms |             1.76 |           28,399 ms* |            0.60* |                              1.0× |
| `blocked_bs64`     | `si-sk-sj`, flat `double*`            |               222 ms |             9.67 |                    — |                — |                              5.5× |
| `fast_ikj`         | `i-k-j`, `double**`                   |               158 ms |             13.6 |              1316 ms |             13.1 |                              7.7× |
| `ikj_k4`           | `i-k-j` + `k×4` unroll                |               127 ms |             16.9 |              1377 ms |             12.5 |                              9.6× |
| `neon_4x4_bs64`    | 4×4 NEON FMA + packed `B`             |          **73.9 ms** |         **29.1** |           **702 ms** |         **25.1** |                         **16.5×** |
| `accelerate_dgemm` | Apple Accelerate (`VECLIB_THREADS=1`) |          **6.35 ms** |        **339.5** |          **56.1 ms** |        **307.1** |                        **191.8×** |

*\*`naive_ijk` at N = 2048 was measured in an earlier `-O2` run.*

---

## 2. OpenCV Comparison

The optional `matmul-opencv` target ([`src/opencv_benchmark.cpp`](./src/opencv_benchmark.cpp), using wrappers in [`src/opencv-gemm.cpp`](./src/opencv-gemm.cpp)) benchmarks OpenCV 4 on `cv::Mat` across dense, rectangular, complex, Gram, and vision workloads, compared against the standalone C++/NEON kernels in [`src/google_benchmark.cpp`](./src/google_benchmark.cpp):

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
