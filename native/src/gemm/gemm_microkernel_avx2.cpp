#include "renorm/gemm_internal.h"
#include <immintrin.h>

namespace renorm
{

void microkernel_8x8_avx2(
    const float* packedA,
    const float* packedB,
    float* C,
    int ldc,
    int K)
{
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6 * ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7 * ldc);

    for (int k = 0; k < K; ++k)
    {
        _mm_prefetch(reinterpret_cast<const char*>(packedB + (k + 8) * 8), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(packedA + (k + 8) * 8), _MM_HINT_T0);

        const __m256 b = _mm256_load_ps(packedB + k * 8);

        // Linearized access: elements are sequential inside the packed panel
        const float* a_ptr = packedA + k * 8;
        const __m256 a0 = _mm256_broadcast_ss(a_ptr + 0);
        const __m256 a1 = _mm256_broadcast_ss(a_ptr + 1);
        const __m256 a2 = _mm256_broadcast_ss(a_ptr + 2);
        const __m256 a3 = _mm256_broadcast_ss(a_ptr + 3);
        const __m256 a4 = _mm256_broadcast_ss(a_ptr + 4);
        const __m256 a5 = _mm256_broadcast_ss(a_ptr + 5);
        const __m256 a6 = _mm256_broadcast_ss(a_ptr + 6);
        const __m256 a7 = _mm256_broadcast_ss(a_ptr + 7);

#if defined(__FMA__)
        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
        c4 = _mm256_fmadd_ps(a4, b, c4);
        c5 = _mm256_fmadd_ps(a5, b, c5);
        c6 = _mm256_fmadd_ps(a6, b, c6);
        c7 = _mm256_fmadd_ps(a7, b, c7);
#else
        c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b));
        c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b));
        c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b));
        c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b));
        c4 = _mm256_add_ps(c4, _mm256_mul_ps(a4, b));
        c5 = _mm256_add_ps(c5, _mm256_mul_ps(a5, b));
        c6 = _mm256_add_ps(c6, _mm256_mul_ps(a6, b));
        c7 = _mm256_add_ps(c7, _mm256_mul_ps(a7, b));
#endif
    }

    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
    _mm256_storeu_ps(C + 4 * ldc, c4);
    _mm256_storeu_ps(C + 5 * ldc, c5);
    _mm256_storeu_ps(C + 6 * ldc, c6);
    _mm256_storeu_ps(C + 7 * ldc, c7);
}

} // namespace renorm