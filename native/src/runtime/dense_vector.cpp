#include "renorm/runtime/dense_vector.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

DenseVector::

DenseVector()

{
}

DenseVector::

DenseVector(

int size)

{
    resize(size);
}

void DenseVector::

resize(

int size)

{
    values_.assign(size,0.0f);
}

float*

DenseVector::

data()

{
    return values_.data();
}

const float*

DenseVector::

data() const

{
    return values_.data();
}

int DenseVector::

size() const

{
    return static_cast<int>(values_.size());
}

void DenseVector::

clear()

{
    std::fill(

        values_.begin(),

        values_.end(),

        0.0f);
}

bool DenseVector::

any() const

{
    for(float v:values_)
    {
        if(v!=0.0f)

            return true;
    }

    return false;
}

void DenseVector::

swap(

DenseVector& other)

{
    values_.swap(other.values_);
}

float&

DenseVector::

operator[](int i)

{
    return values_[i];
}

const float&

DenseVector::

operator[](int i) const

{
    return values_[i];
}

}
}