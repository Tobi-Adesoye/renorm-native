#include "renorm/gemm.h"

#include <algorithm>
#include <cstddef>

namespace renorm
{

void gemm_scalar(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    ctx.increment_kernel_calls();

    std::fill(
        C,
        C + static_cast<std::size_t>(M) * N,
        0.0f
    );

    for (int i = 0; i < M; ++i)
    {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;

        for (int k = 0; k < K; ++k)
        {
            const float a = a_row[k];
            const float* b_row = B + k * N;

            for (int j = 0; j < N; ++j)
            {
                c_row[j] += a * b_row[j];
            }
        }
    }
}

}