#pragma once

#include <algorithm>
#include <cmath>

namespace renorm
{

//--------------------------------------------------------------
// Reference GEMM
//
// Computes:
//
// C = A × B
//
// A: M x K
// B: K x N
// C: M x N
//
// This is the trusted scalar implementation used ONLY for
// validation. It is intentionally simple and not optimized.
//--------------------------------------------------------------

inline void reference_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    // Zero output

    std::fill(C, C + (M * N), 0.0f);

    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            const float a = A[i * K + k];

            for (int j = 0; j < N; ++j)
            {
                C[i * N + j] +=
                    a * B[k * N + j];
            }
        }
    }
}

//--------------------------------------------------------------
// Maximum Absolute Error
//--------------------------------------------------------------

inline float max_absolute_error(
    const float* reference,
    const float* candidate,
    int size)
{
    float max_error = 0.0f;

    for (int i = 0; i < size; ++i)
    {
        float error =
            std::fabs(
                reference[i] -
                candidate[i]);

        if (error > max_error)
            max_error = error;
    }

    return max_error;
}

//--------------------------------------------------------------
// Maximum Relative Error
//--------------------------------------------------------------

inline float max_relative_error(
    const float* reference,
    const float* candidate,
    int size)
{
    float max_error = 0.0f;

    constexpr float epsilon = 1e-12f;

    for (int i = 0; i < size; ++i)
    {
        float denom =
            std::max(
                std::fabs(reference[i]),
                epsilon);

        float error =
            std::fabs(
                reference[i] -
                candidate[i]) / denom;

        if (error > max_error)
            max_error = error;
    }

    return max_error;
}

//--------------------------------------------------------------
// Root Mean Square Error
//--------------------------------------------------------------

inline float rmse(
    const float* reference,
    const float* candidate,
    int size)
{
    double sum = 0.0;

    for (int i = 0; i < size; ++i)
    {
        double diff =
            static_cast<double>(reference[i]) -
            static_cast<double>(candidate[i]);

        sum += diff * diff;
    }

    return static_cast<float>(
        std::sqrt(sum / static_cast<double>(size)));
}

//--------------------------------------------------------------
// Validation Result
//--------------------------------------------------------------

struct ValidationResult
{
    float max_abs_error;
    float max_rel_error;
    float rmse;
    bool passed;
};

//--------------------------------------------------------------
// Compare Results
//--------------------------------------------------------------

inline ValidationResult validate_gemm(
    const float* reference,
    const float* candidate,
    int size,
    float tolerance = 1e-4f)
{
    ValidationResult result;

    result.max_abs_error =
        max_absolute_error(
            reference,
            candidate,
            size);

    result.max_rel_error =
        max_relative_error(
            reference,
            candidate,
            size);

    result.rmse =
        rmse(
            reference,
            candidate,
            size);

    result.passed =
        result.max_abs_error <= tolerance &&
        result.max_rel_error <= tolerance;

    return result;
}

} // namespace renorm