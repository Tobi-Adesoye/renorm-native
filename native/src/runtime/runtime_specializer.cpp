#include "renorm/runtime/runtime_specializer.h"

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void RuntimeSpecializer::specialize(
    ExecutionContext& context) const
{
    const Backend backend =
        detect_backend();

    switch (backend)
    {
        case Backend::Scalar:

            specialize_scalar(context);
            break;

        case Backend::SSE42:

            specialize_sse42(context);
            break;

        case Backend::AVX2:

            specialize_avx2(context);
            break;

        case Backend::AVX512:

            specialize_avx512(context);
            break;

        case Backend::NEON:

            specialize_neon(context);
            break;

        case Backend::SVE:

            specialize_sve(context);
            break;

        case Backend::AppleAMX:

            specialize_amx(context);
            break;

        case Backend::CUDA:

            specialize_cuda(context);
            break;

        case Backend::HIP:

            specialize_hip(context);
            break;
    }
}

//==============================================================

RuntimeSpecializer::Backend
RuntimeSpecializer::detect_backend() const
{

#if defined(__AVX512F__)

    return Backend::AVX512;

#elif defined(__AVX2__)

    return Backend::AVX2;

#elif defined(__SSE4_2__)

    return Backend::SSE42;

#elif defined(__ARM_FEATURE_SVE)

    return Backend::SVE;

#elif defined(__ARM_NEON)

    return Backend::NEON;

#else

    return Backend::Scalar;

#endif
}

//==============================================================

void RuntimeSpecializer::specialize_scalar(
    ExecutionContext& context) const
{
    context.vectorWidth = 1;

    context.backendName = "Scalar";
}

//==============================================================

void RuntimeSpecializer::specialize_sse42(
    ExecutionContext& context) const
{
    context.vectorWidth = 4;

    context.backendName = "SSE4.2";

    context.simdExecutionEnabled = true;
}

//==============================================================

void RuntimeSpecializer::specialize_avx2(
    ExecutionContext& context) const
{
    context.vectorWidth = 8;

    context.backendName = "AVX2";

    context.simdExecutionEnabled = true;

    context.prefetchEnabled = true;

    context.vectorPlanEnabled = true;
}

//==============================================================

void RuntimeSpecializer::specialize_avx512(
    ExecutionContext& context) const
{
    context.vectorWidth = 16;

    context.backendName = "AVX512";

    context.simdExecutionEnabled = true;

    context.prefetchEnabled = true;

    context.vectorPlanEnabled = true;
}

//==============================================================

void RuntimeSpecializer::specialize_neon(
    ExecutionContext& context) const
{
    context.vectorWidth = 4;

    context.backendName = "NEON";

    context.simdExecutionEnabled = true;
}

//==============================================================

void RuntimeSpecializer::specialize_sve(
    ExecutionContext& context) const
{
    context.vectorWidth = 16;

    context.backendName = "SVE";

    context.simdExecutionEnabled = true;
}

//==============================================================

void RuntimeSpecializer::specialize_amx(
    ExecutionContext& context) const
{
    context.backendName = "AppleAMX";

    context.matrixAccelerationEnabled = true;
}

//==============================================================

void RuntimeSpecializer::specialize_cuda(
    ExecutionContext& context) const
{
    context.backendName = "CUDA";

    context.gpuExecutionEnabled = true;
}

//==============================================================

void RuntimeSpecializer::specialize_hip(
    ExecutionContext& context) const
{
    context.backendName = "HIP";

    context.gpuExecutionEnabled = true;
}

} // namespace runtime
} // namespace renorm