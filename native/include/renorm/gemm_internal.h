#pragma once

#include "renorm/kernel_context.h"
#include "renorm/worker_context.h"

namespace renorm
{

//==============================================================
// Internal GEMM Worker
//==============================================================

void gemm_worker(
    WorkerContext& worker,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K);

//==============================================================
// Scalar Kernel
//==============================================================

void gemm_scalar(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K);

//==============================================================
// AVX2 Kernel (Parallel Worker Version)
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
// Sequential Compatibility Wrapper
//==============================================================

void gemm_avx2(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K);

//==============================================================
// Parallel Driver
//==============================================================

void gemm_parallel(
    KernelContext& ctx,
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K);

//==============================================================
// Microkernel
//==============================================================

void microkernel_8x8_avx2(
    const float* packedA,
    const float* packedB,
    float* C,
    int ldc,
    int K);

} // namespace renorm