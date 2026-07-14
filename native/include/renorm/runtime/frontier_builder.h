#pragma once

#include "renorm/runtime/frontier.h"
#include "renorm/runtime/compiled_graph.h"
#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// FrontierBuilder
//
// Builds the runtime frontier from a compiled graph.
//
// Supports:
//
//  • Legacy API
//  • ExecutionContext API
//==============================================================

class FrontierBuilder
{
public:

    //----------------------------------------------------------
    // Legacy interface
    //----------------------------------------------------------

    Frontier build(
        const CompiledGraph& compiled,
        NodeId start,
        NodeId goal) const;

    //----------------------------------------------------------
    // New runtime interface
    //----------------------------------------------------------

    void build(
        ExecutionContext& context,
        NodeId start,
        NodeId goal) const;

private:

    void initialize_frontier(
        Frontier& frontier,
        const CompiledGraph& compiled,
        NodeId start,
        NodeId goal) const;
};

}
}