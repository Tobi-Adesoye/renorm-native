#pragma once

#include "renorm/runtime/compiled_graph.h"

namespace renorm
{
namespace runtime
{

class GraphMatrixBuilder
{
public:

    //----------------------------------------------------------
    // Build every matrix representation required by runtime.
    //----------------------------------------------------------

    void build(
        const Graph& graph,
        CompiledGraph& compiled) const;
};

}
}