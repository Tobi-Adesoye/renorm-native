#include "renorm/runtime/dense_matrix_builder.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

DenseMatrix DenseMatrixBuilder::build(
    const Graph& graph)
{
    const std::size_t n =
        graph.node_count();

    DenseMatrix matrix;

    //----------------------------------------------------------
    // Allocate NxN adjacency matrix
    //----------------------------------------------------------

    matrix.resize(n);

    //----------------------------------------------------------
    // Initialize to zero
    //----------------------------------------------------------

    std::fill(
        matrix.data.begin(),
        matrix.data.end(),
        0.0f);

    //----------------------------------------------------------
    // Build adjacency matrix
    //----------------------------------------------------------

    for (NodeId u = 0; u < n; ++u)
    {
        for (NodeId v : graph.neighbors(u))
        {
            matrix(u, v) = 1.0f;
        }
    }

    return matrix;
}

}
}