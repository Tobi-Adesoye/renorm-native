#include "renorm/gemm.h"
#include "renorm/gemm_dispatch.h"

namespace renorm
{

void gemm(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    gemm_dispatch(
        ctx,
        A,
        B,
        C,
        M,
        N,
        K);
}

}