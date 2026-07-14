#include "renorm/gemm.h"

#include <algorithm>
#include <cstddef>

namespace renorm
{

namespace
{
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 64;
}

void gemm_blocked(
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

    for (int ii = 0; ii < M; ii += BLOCK_M)
    {
        const int i_end = std::min(ii + BLOCK_M, M);

        for (int kk = 0; kk < K; kk += BLOCK_K)
        {
            const int k_end = std::min(kk + BLOCK_K, K);

            for (int jj = 0; jj < N; jj += BLOCK_N)
            {
                const int j_end = std::min(jj + BLOCK_N, N);

                for (int i = ii; i < i_end; ++i)
                {
                    float* c_row = C + i * N;
                    const float* a_row = A + i * K;

                    for (int k = kk; k < k_end; ++k)
                    {
                        const float a = a_row[k];
                        const float* b_row = B + k * N;

                        for (int j = jj; j < j_end; ++j)
                        {
                            c_row[j] += a * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

}