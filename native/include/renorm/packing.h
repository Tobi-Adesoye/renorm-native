#pragma once

namespace renorm
{

void pack_a(
    const float* src,
    float* dst,
    int rows,
    int cols,
    int lda);

void pack_b(
    const float* src,
    float* dst,
    int rows,
    int cols,
    int lda);

} // namespace renorm