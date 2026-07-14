#pragma once

#include <cstddef>

namespace renorm
{
namespace runtime
{

class ExecutionContext;

//==============================================================
//
// RuntimeSpecializer
//
// Converts the generic execution plan into a hardware-specific
// execution plan.
//
// Supported targets:
//
//  • Scalar
//  • SSE4.2
//  • AVX2
//  • AVX-512
//  • ARM NEON
//  • ARM SVE
//  • Apple AMX
//  • CUDA
//  • HIP
//
//==============================================================

class RuntimeSpecializer
{
public:

    enum class Backend
    {
        Scalar,

        SSE42,

        AVX2,

        AVX512,

        NEON,

        SVE,

        AppleAMX,

        CUDA,

        HIP
    };

public:

    RuntimeSpecializer() = default;

    //----------------------------------------------------------

    void specialize(
        ExecutionContext& context) const;

private:

    Backend detect_backend() const;

    void specialize_scalar(
        ExecutionContext& context) const;

    void specialize_sse42(
        ExecutionContext& context) const;

    void specialize_avx2(
        ExecutionContext& context) const;

    void specialize_avx512(
        ExecutionContext& context) const;

    void specialize_neon(
        ExecutionContext& context) const;

    void specialize_sve(
        ExecutionContext& context) const;

    void specialize_amx(
        ExecutionContext& context) const;

    void specialize_cuda(
        ExecutionContext& context) const;

    void specialize_hip(
        ExecutionContext& context) const;
};

}
}