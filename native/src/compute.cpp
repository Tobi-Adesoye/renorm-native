#include "renorm/compute.h"
#include "renorm/gemm_dispatch.h"

#include <stdexcept>

namespace renorm
{

Tensor matmul(
    KernelContext& ctx,
    const Tensor& A,
    const Tensor& B)
{
    //----------------------------------------------------------
    // Validate tensors
    //----------------------------------------------------------

    if (A.dims() != 2 || B.dims() != 2)
    {
        throw std::runtime_error(
            "matmul requires two 2D tensors."
        );
    }

    const std::size_t M = A.shape()[0];
    const std::size_t K = A.shape()[1];

    if (B.shape()[0] != K)
    {
        throw std::runtime_error(
            "Invalid matrix dimensions."
        );
    }

    const std::size_t N = B.shape()[1];

    //----------------------------------------------------------
    // Allocate output tensor
    //----------------------------------------------------------

    Tensor C({ M, N });

    //----------------------------------------------------------
    // Runtime kernel dispatch
    //
    // Automatically selects:
    //
    //  • Scalar
    //  • Blocked
    //  • AVX2
    //
    // depending on CPU capabilities.
    //----------------------------------------------------------

    gemm_dispatch(
        ctx,
        A.data(),
        B.data(),
        C.data(),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K)
    );

    return C;
}

Tensor linear(
    KernelContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias)
{
    Tensor output =
        matmul(
            ctx,
            input,
            weight
        );

    if (!bias)
    {
        return output;
    }

    if (bias->dims() != 1)
    {
        throw std::runtime_error(
            "Bias must be 1D."
        );
    }

    const std::size_t rows =
        output.shape()[0];

    const std::size_t cols =
        output.shape()[1];

    float* out =
        output.data();

    const float* b =
        bias->data();

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            out[i * cols + j] += b[j];
        }
    }

    return output;
}

} // namespace renorm