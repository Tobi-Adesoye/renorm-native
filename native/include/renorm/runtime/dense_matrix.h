#pragma once

#include <vector>
#include <cstddef>
#include <cassert>

namespace renorm
{
namespace runtime
{

class DenseMatrix
{
public:

    DenseMatrix() = default;

    explicit DenseMatrix(
        std::size_t dimension)
    {
        resize(dimension);
    }

    //----------------------------------------------------------
    // Resize
    //----------------------------------------------------------

    void resize(
        std::size_t dimension)
    {
        m_dimension = dimension;

        data.assign(
            dimension * dimension,
            0.0f);
    }

    //----------------------------------------------------------
    // Dimension
    //----------------------------------------------------------

    std::size_t dimension() const
    {
        return m_dimension;
    }

    std::size_t rows() const
    {
        return m_dimension;
    }

    std::size_t cols() const
    {
        return m_dimension;
    }

    //----------------------------------------------------------
    // Element access
    //----------------------------------------------------------

    float& operator()(
        std::size_t row,
        std::size_t column)
    {
        assert(row < m_dimension);
        assert(column < m_dimension);

        return data[
            row * m_dimension +
            column];
    }

    const float& operator()(
        std::size_t row,
        std::size_t column) const
    {
        assert(row < m_dimension);
        assert(column < m_dimension);

        return data[
            row * m_dimension +
            column];
    }

public:

    //----------------------------------------------------------
    // Raw contiguous storage
    //----------------------------------------------------------

    std::vector<float> data;

private:

    std::size_t m_dimension = 0;
};

}
}