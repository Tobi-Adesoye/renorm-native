#include "packing_internal.h"
#include <cstring>
#include <algorithm>

namespace renorm
{

// -----------------------------------------------------------------------------
// pack_a_panel: Packs matrix A into contiguous blocks of size MR x kc (8 x kc)
// If rows (mc) is not a multiple of MR, the remainder is zero-padded.
// -----------------------------------------------------------------------------
void pack_a_panel(
    const float* src, 
    float* dst, 
    int mc, 
    int kc, 
    int lda)
{
    // Process full MR-sized panels
    int i = 0;
    for (; i <= mc - MR; i += MR)
    {
        const float* src_row = src + i * lda;
        for (int k = 0; k < kc; ++k)
        {
            dst[0] = src_row[0 * lda + k];
            dst[1] = src_row[1 * lda + k];
            dst[2] = src_row[2 * lda + k];
            dst[3] = src_row[3 * lda + k];
            dst[4] = src_row[4 * lda + k];
            dst[5] = src_row[5 * lda + k];
            dst[6] = src_row[6 * lda + k];
            dst[7] = src_row[7 * lda + k];
            dst += MR;
        }
    }

    // Handle any partial edge row remainder (Zero-padded to preserve layout invariants)
    if (i < mc)
    {
        int remainder_rows = mc - i;
        const float* src_row = src + i * lda;

        for (int k = 0; k < kc; ++k)
        {
            int r = 0;
            for (; r < remainder_rows; ++r)
            {
                dst[r] = src_row[r * lda + k];
            }
            for (; r < MR; ++r)
            {
                dst[r] = 0.0f; // Padding
            }
            dst += MR;
        }
    }
}

// -----------------------------------------------------------------------------
// pack_b_panel: Packs matrix B into contiguous blocks of size kc x NR (kc x 8)
// If cols (nc) is not a multiple of NR, the remainder is zero-padded.
// -----------------------------------------------------------------------------
void pack_b_panel(
    const float* src, 
    float* dst, 
    int kc, 
    int nc, 
    int lda)
{
    // Process full NR-sized panels
    int j = 0;
    for (; j <= nc - NR; j += NR)
    {
        for (int k = 0; k < kc; ++k)
        {
            const float* src_row = src + k * lda + j;
            std::memcpy(dst, src_row, NR * sizeof(float));
            dst += NR;
        }
    }

    // Handle any partial edge column remainder (Zero-padded to preserve layout invariants)
    if (j < nc)
    {
        int remainder_cols = nc - j;
        for (int k = 0; k < kc; ++k)
        {
            const float* src_row = src + k * lda + j;
            int c = 0;
            for (; c < remainder_cols; ++c)
            {
                dst[c] = src_row[c];
            }
            for (; c < NR; ++c)
            {
                dst[c] = 0.0f; // Padding
            }
            dst += NR;
        }
    }
}

} // namespace renorm