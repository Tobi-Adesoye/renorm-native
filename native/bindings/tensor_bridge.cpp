#pragma once

#include <torch/extension.h>

#include "renorm/tensor.h"

namespace renorm
{

Tensor tensor_from_torch(const torch::Tensor& tensor);

torch::Tensor tensor_to_torch(
    const Tensor& tensor
);

}