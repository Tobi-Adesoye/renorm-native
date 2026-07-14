#include "renorm/packing.h"
#include <algorithm>
#include <cstring>

namespace renorm
{

void pack_a(
    const float* src,
    float* dst,
    int rows,
    int cols,
    int lda)
{
    for (int i = 0; i < rows; ++i)
    {
        std::memcpy(dst + i * cols, src + i * lda, cols * sizeof(float));
    }
}

void pack_b(
    const float* src,
    float* dst,
    int rows,
    int cols,
    int lda)
{
    for (int i = 0; i < rows; ++i)
    {
        std::memcpy(dst + i * cols, src + i * lda, cols * sizeof(float));
    }
}

} // namespace renorm