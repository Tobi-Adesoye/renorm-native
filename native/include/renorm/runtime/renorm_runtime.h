#pragma once

#include "renorm/graph.h"

#include "renorm/runtime/compiled_graph.h"
#include "renorm/runtime/runtime_result.h"

namespace renorm
{
namespace runtime
{

class RenormRuntime
{
public:

    //----------------------------------------------------------
    // Convenience execution
    //----------------------------------------------------------

    RuntimeResult execute(
        const Graph& graph);

    RuntimeResult execute(
        const CompiledGraph& compiled);

    //----------------------------------------------------------
    // Explicit start/goal execution
    //----------------------------------------------------------

    RuntimeResult execute(
        const Graph& graph,
        NodeId start,
        NodeId goal);

    RuntimeResult execute(
        const CompiledGraph& compiled,
        NodeId start,
        NodeId goal);
};

} // namespace runtime
} // namespace renorm