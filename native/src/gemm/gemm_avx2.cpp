#include "renorm/gemm.h"
#include "renorm/gemm_internal.h"
#include "renorm/kernel_context.h"
#include "renorm/packing.h"
#include "renorm/worker_context.h"
#include "renorm/tuner.h"

// Include the internal panel packing declarations (defines MR and NR)
#include "packing_internal.h" 

namespace renorm
{

void microkernel_8x8_avx2(
    const float* packedA,
    const float* packedB,
    float* C,
    int ldc,
    int K);

// REMOVED: constexpr int MR = 8;  <-- These are now supplied directly by packing_internal.h
// REMOVED: constexpr int NR = 8;

static void gemm_worker(
    WorkerContext& worker,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    const BlockConfig& cfg)
{
    const int MC = cfg.MC;
    const int NC = cfg.NC;
    const int KC = cfg.KC;

    const std::size_t RequiredWorkspace = sizeof(float) * (MC * KC + KC * NC);

    if (worker.workspace.capacity() < RequiredWorkspace)
    {
        worker.workspace.reserve(RequiredWorkspace);
    }

    worker.workspace.reset();

    float* packedA = worker.workspace.allocate<float>(MC * KC);
    float* packedB = worker.workspace.allocate<float>(KC * NC);

    for (int jc = 0; jc < N; jc += NC)
    {
        const int nc = std::min(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC)
        {
            const int kc = std::min(KC, K - pc);

            pack_b_panel(B + pc * N + jc, packedB, kc, nc, N);

            for (int ic = 0; ic < M; ic += MC)
            {
                const int mc = std::min(MC, M - ic);
                
                pack_a_panel(A + ic * K + pc, packedA, mc, kc, K);

                const int mcAligned = (mc / MR) * MR;
                const int ncAligned = (nc / NR) * NR;

                // -------------------------------------------------------------
                // 1. Full Register Tiles (Panel-To-Panel Traversal)
                // -------------------------------------------------------------
                for (int i = 0; i < mcAligned; i += MR)
                {
                    for (int j = 0; j < ncAligned; j += NR)
                    {
                        if (kc >= 64)
                        {
                            const int nextPanelA = (i / MR) + 1;
                            const int nextPanelB = (j / NR) + 1;
                            _mm_prefetch(reinterpret_cast<const char*>(packedA + nextPanelA * kc * MR), _MM_HINT_T0);
                            _mm_prefetch(reinterpret_cast<const char*>(packedB + nextPanelB * kc * NR), _MM_HINT_T0);
                        }

                        microkernel_8x8_avx2(
                            packedA + (i / MR) * kc * MR,
                            packedB + (j / NR) * kc * NR,
                            C + (ic + i) * N + (jc + j),
                            N,
                            kc);
                    }
                }

                // -------------------------------------------------------------
                // 2. Right Edge Scalar Fallback (Handling Remainder Columns)
                // -------------------------------------------------------------
                for (int i = 0; i < mcAligned; i += MR)
                {
                    for (int j = ncAligned; j < nc; ++j)
                    {
                        float* c = C + (ic + i) * N + (jc + j);
                        const int panelB = j / NR;
                        const int colInB = j % NR;

                        for (int k = 0; k < kc; ++k)
                        {
                            const float b = packedB[panelB * kc * NR + k * NR + colInB];
                            const float* a_ptr = packedA + (i / MR) * kc * MR + k * MR;
                            
                            c[0 * N] += a_ptr[0] * b;
                            c[1 * N] += a_ptr[1] * b;
                            c[2 * N] += a_ptr[2] * b;
                            c[3 * N] += a_ptr[3] * b;
                            c[4 * N] += a_ptr[4] * b;
                            c[5 * N] += a_ptr[5] * b;
                            c[6 * N] += a_ptr[6] * b;
                            c[7 * N] += a_ptr[7] * b;
                        }
                    }
                }

                // -------------------------------------------------------------
                // 3. Bottom Edge Scalar Fallback (Handling Remainder Rows)
                // -------------------------------------------------------------
                for (int i = mcAligned; i < mc; ++i)
                {
                    for (int j = 0; j < nc; ++j)
                    {
                        float* c = C + (ic + i) * N + (jc + j);
                        const int panelB = j / NR;
                        const int colInB = j % NR;

                        for (int k = 0; k < kc; ++k)
                        {
                            const float av = A[(ic + i) * K + (pc + k)];
                            const float b = packedB[panelB * kc * NR + k * NR + colInB];
                            *c += av * b;
                        }
                    }
                }
            }
        }
    }
}

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
    int K)
{
    ctx.increment_kernel_calls();
    
    const BlockConfig& cfg = get_block_config();

    gemm_worker(
        worker,
        A + rowBegin * K,
        B,
        C + rowBegin * N,
        rowEnd - rowBegin,
        N,
        K,
        cfg);
}

void gemm_avx2(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    WorkerContext worker;
    gemm_avx2(ctx, worker, A, B, C, 0, M, M, N, K);
}

} // namespace renorm