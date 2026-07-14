#pragma once

#include "kernel_context.h"

namespace renorm
{

// Public GEMM API.
// Internally this dispatches to the best available implementation
// (Scalar, Blocked, AVX2, ...).

void gemm(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K);

}