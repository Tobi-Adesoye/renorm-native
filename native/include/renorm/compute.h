#pragma once

#include "tensor.h"
#include "kernel_context.h"

namespace renorm
{

Tensor matmul(
    KernelContext& context,
    const Tensor& a,
    const Tensor& b
);

Tensor linear(
    KernelContext& context,
    const Tensor& input,
    const Tensor& weight,
    const Tensor* bias
);

}