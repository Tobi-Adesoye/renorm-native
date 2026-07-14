#pragma once

#include "renorm/graph.h"
#include "renorm/runtime/dense_matrix.h"

namespace renorm
{
namespace runtime
{

class DenseMatrixBuilder
{
public:

    static DenseMatrix build(
        const Graph& graph);
};

}
}