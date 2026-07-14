#pragma once

#include <vector>

namespace renorm
{
namespace runtime
{

class DenseVector
{
public:

    DenseVector();

    explicit DenseVector(
        int size);

    void resize(
        int size);

    float*

    data();

    const float*

    data() const;

    int size() const;

    void clear();

    bool any() const;

    void swap(
        DenseVector& other);

    float& operator[](int i);

    const float& operator[](int i) const;

private:

    std::vector<float> values_;

};

}
}