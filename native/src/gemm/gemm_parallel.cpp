#include "renorm/gemm_internal.h"
#include "renorm/kernel_context.h"
#include "renorm/scheduler.h"

#include <algorithm>

namespace renorm
{

//==============================================================
// External AVX2 Worker
//==============================================================

void gemm_avx2(
    KernelContext& ctx,
    WorkerContext& worker,
    const float* A,
    const float* B,
    float* C,
    int rowBegin,
    int rowEnd,
    int M,
    int N,
    int K);

//==============================================================
// Parallel Entry Point
//==============================================================

void gemm_parallel(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    //----------------------------------------------------------
    // Small Problems
    //----------------------------------------------------------

    constexpr int ParallelThreshold = 128;

    if (M < ParallelThreshold)
    {
        WorkerContext worker;

        gemm_avx2(
            ctx,
            worker,
            A,
            B,
            C,
            0,
            M,
            M,
            N,
            K);

        return;
    }

    //----------------------------------------------------------
    // Parallel Row Blocking
    //----------------------------------------------------------

    constexpr int GrainRows = 64;

    parallel_for(
        0,
        M,
        GrainRows,

        [&](WorkerContext& worker,
            int rowBegin,
            int rowEnd)
        {
            gemm_avx2(
                ctx,
                worker,
                A,
                B,
                C,
                rowBegin,
                rowEnd,
                M,
                N,
                K);
        });
}

} // namespace renorm