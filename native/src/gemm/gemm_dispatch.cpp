#include "renorm/gemm_dispatch.h"

#include "renorm/gemm_internal.h"
#include "renorm/cpu_features.h"

namespace renorm
{

#ifdef RENORM_ENABLE_AVX512

void gemm_avx512(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K);

#endif

//==============================================================
// Runtime Dispatcher
//==============================================================

void gemm_dispatch(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    //----------------------------------------------------------
    // Invalid dimensions
    //----------------------------------------------------------

    if (M <= 0 || N <= 0 || K <= 0)
    {
        return;
    }

#ifdef RENORM_ENABLE_AVX512

    //----------------------------------------------------------
    // AVX-512
    //----------------------------------------------------------

    if (has_avx512())
    {
        gemm_avx512(
            ctx,
            A,
            B,
            C,
            M,
            N,
            K);

        return;
    }

#endif

    //----------------------------------------------------------
    // AVX2
    //----------------------------------------------------------

    if (has_avx2())
    {
        gemm_parallel(
            ctx,
            A,
            B,
            C,
            M,
            N,
            K);

        return;
    }

    //----------------------------------------------------------
    // Scalar Fallback
    //----------------------------------------------------------

    gemm_scalar(
        ctx,
        A,
        B,
        C,
        M,
        N,
        K);
}

} // namespace renorm