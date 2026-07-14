#pragma once

#include "renorm/graph.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// Execution instruction
//
// Immutable runtime instruction emitted by the compiler.
//--------------------------------------------------------------

struct ExecutionInstruction
{
    NodeId from = INVALID_NODE;
    NodeId to   = INVALID_NODE;

    float weight = 0.0f;
};

//--------------------------------------------------------------
// CompiledBlock
//
// Final runtime execution unit.
//
// The runtime executes CompiledBlocks directly rather than
// traversing graph adjacency lists.
//--------------------------------------------------------------

struct CompiledBlock
{
    //----------------------------------------------------------
    // Runtime ID
    //----------------------------------------------------------

    std::size_t id = 0;

    //----------------------------------------------------------
    // Nodes owned by this block
    //----------------------------------------------------------

    std::vector<NodeId> nodes;

    //----------------------------------------------------------
    // Compiled execution program
    //----------------------------------------------------------

    std::vector<ExecutionInstruction> program;

    //----------------------------------------------------------
    // Block graph
    //----------------------------------------------------------

    std::vector<std::size_t> outgoingBlocks;
    std::vector<std::size_t> incomingBlocks;

    //----------------------------------------------------------
    // Metadata
    //----------------------------------------------------------

    std::size_t instructionCount = 0;

    std::size_t nonZeros = 0;

    float density = 0.0f;

    //----------------------------------------------------------
    // Execution statistics
    //----------------------------------------------------------

    bool executable = true;
};

} // namespace runtime
} // namespace renorm