#include "renorm/tensor.h"

#include "renorm/memory.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace renorm
{

Tensor::Tensor()
    : element_count_(0),
      data_(nullptr)
{
}

Tensor::Tensor(const std::vector<std::size_t>& shape)
    : shape_(shape),
      element_count_(0),
      data_(nullptr)
{
    allocate();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      element_count_(other.element_count_),
      data_(other.data_)
{
    other.data_ = nullptr;
    other.element_count_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept
{
    if (this != &other)
    {
        release();

        shape_ = std::move(other.shape_);
        element_count_ = other.element_count_;
        data_ = other.data_;

        other.data_ = nullptr;
        other.element_count_ = 0;
    }

    return *this;
}

Tensor::~Tensor()
{
    release();
}

void Tensor::resize(const std::vector<std::size_t>& shape)
{
    release();

    shape_ = shape;

    allocate();
}

void Tensor::allocate()
{
    element_count_ = 1;

    for (std::size_t d : shape_)
    {
        element_count_ *= d;
    }

    if (element_count_ == 0)
        return;

    data_ = static_cast<float*>(
        aligned_alloc(
            sizeof(float) * element_count_
        )
    );

    if (!data_)
    {
        throw std::bad_alloc();
    }

    std::memset(
        data_,
        0,
        sizeof(float) * element_count_
    );
}

void Tensor::release()
{
    if (data_)
    {
        aligned_free(data_);
        data_ = nullptr;
    }

    element_count_ = 0;
}

float* Tensor::data()
{
    return data_;
}

const float* Tensor::data() const
{
    return data_;
}

std::size_t Tensor::size() const
{
    return element_count_;
}

const std::vector<std::size_t>& Tensor::shape() const
{
    return shape_;
}

std::size_t Tensor::dims() const
{
    return shape_.size();
}

bool Tensor::empty() const
{
    return element_count_ == 0;
}

void Tensor::fill(float value)
{
    std::fill(
        data_,
        data_ + element_count_,
        value
    );
}

}