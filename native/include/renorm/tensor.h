#pragma once

#include <cstddef>
#include <vector>

namespace renorm {

class Tensor
{
public:

    Tensor();

    explicit Tensor(const std::vector<std::size_t>& shape);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    ~Tensor();

    void resize(const std::vector<std::size_t>& shape);

    float* data();
    const float* data() const;

    std::size_t size() const;

    const std::vector<std::size_t>& shape() const;

    std::size_t dims() const;

    bool empty() const;

    void fill(float value);

private:

    void allocate();
    void release();

private:

    std::vector<std::size_t> shape_;

    std::size_t element_count_;

    float* data_;

};

}