#pragma once

#include "renorm/runtime/frontier.h"
#include "renorm/runtime/compiled_graph.h"

namespace renorm
{
namespace runtime
{

class FrontierExecutor
{
public:

    virtual ~FrontierExecutor() = default;

    //----------------------------------------------------------
    // Execute ONE frontier iteration.
    //----------------------------------------------------------

    virtual void execute_iteration(
        Frontier& frontier,
        const CompiledGraph& graph) = 0;

    //----------------------------------------------------------
    // Execute until completion.
    //----------------------------------------------------------

    virtual void execute(
        Frontier& frontier,
        const CompiledGraph& graph);

};

}
}