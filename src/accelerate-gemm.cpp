#include "matrix-multiply.hpp"
#include <cassert>

#if defined(MATMUL_HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(MATMUL_HAVE_CBLAS)
#if defined(__has_include)
#if __has_include(<cblas.h>)
#include <cblas.h>
#elif __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#endif
#else
#include <cblas.h>
#endif
#endif

namespace calculate_matrix
{
#if defined(MATMUL_HAVE_ACCELERATE) || defined(MATMUL_HAVE_CBLAS)
    // Apple Accelerate (macOS) or OpenBLAS/CBLAS (Linux/Windows). C = 1.0 * A * B + 0.0 * C.
    void accelerate_dgemm_matrix_multiply(double *C, const double *A, const double *B, int N)
    {
        assert(N >= 0);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);
    }
#endif
}
