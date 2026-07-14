#pragma once

#include "renorm/graph.h"
#include "renorm/runtime/blocked_matrix.h"

namespace renorm
{
namespace runtime
{

class BlockedMatrixBuilder
{
public:

    static BlockedMatrix build(
        const Graph& graph);
};

}
}