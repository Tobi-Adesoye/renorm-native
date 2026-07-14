#include "renorm/runtime/block_fusion.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void BlockFusion::fuse(
    ExecutionContext& context)
{
    if (!context.graph)
        return;

    auto& blocks =
        context.graph->compiledBlocks;

    if (blocks.size() < 2)
        return;

    //----------------------------------------------------------

    std::size_t i = 0;

    while (i + 1 < blocks.size())
    {
        if (can_fuse(
                blocks[i],
                blocks[i + 1]))
        {
            merge(
                context,
                i,
                i + 1);

            blocks.erase(
                blocks.begin() + i + 1);

            continue;
        }

        ++i;
    }
}

//==============================================================

bool BlockFusion::can_fuse(
    const CompiledBlock& left,
    const CompiledBlock& right) const
{
    //----------------------------------------------------------
    // Single dependency
    //----------------------------------------------------------

    if (left.outgoingBlocks.size() != 1)
        return false;

    //----------------------------------------------------------

    if (right.incomingBlocks.size() != 1)
        return false;

    //----------------------------------------------------------
    // Avoid huge blocks
    //----------------------------------------------------------

    if (left.program.size() +
        right.program.size() >
        512)
        return false;

    return true;
}

//==============================================================

void BlockFusion::merge(
    ExecutionContext& context,
    std::size_t leftIndex,
    std::size_t rightIndex)
{
    auto& blocks =
        context.graph->compiledBlocks;

    auto& left =
        blocks[leftIndex];

    auto& right =
        blocks[rightIndex];

    //----------------------------------------------------------
    // Merge nodes
    //----------------------------------------------------------

    left.nodes.insert(
        left.nodes.end(),
        right.nodes.begin(),
        right.nodes.end());

    //----------------------------------------------------------
    // Merge program
    //----------------------------------------------------------

    left.program.insert(
        left.program.end(),
        right.program.begin(),
        right.program.end());

    //----------------------------------------------------------
    // Merge outgoing
    //----------------------------------------------------------

    left.outgoingBlocks =
        right.outgoingBlocks;

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    left.instructionCount =
        left.program.size();

    left.edgeCount =
        left.program.size();

    left.density =
        (left.density +
         right.density) * 0.5f;

    //----------------------------------------------------------
    // Runtime statistics
    //----------------------------------------------------------

    ++context.executedBlocks;

    context.executionEfficiency *=
        1.02f;

    if (context.executionEfficiency > 1.0f)
        context.executionEfficiency = 1.0f;
}

}
}