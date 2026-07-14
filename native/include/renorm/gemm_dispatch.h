#pragma once

#include "kernel_context.h"

namespace renorm
{

void gemm_dispatch(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K);

}