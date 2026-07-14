#include "renorm/runtime/instruction_fusion.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void InstructionFusion::fuse(
    ExecutionContext& context)
{
    if (!context.graph)
        return;

    //----------------------------------------------------------
    // Process every compiled block
    //----------------------------------------------------------

    for (std::size_t block = 0;
         block < context.graph->compiledBlocks.size();
         ++block)
    {
        fuse_block(
            context,
            block);
    }
}

//==============================================================

void InstructionFusion::fuse_block(
    ExecutionContext& context,
    std::size_t blockId)
{
    auto& block =
        context.graph->compiledBlocks[
            blockId];

    if (block.instructions.size() < 2)
        return;

    //----------------------------------------------------------
    // Greedy adjacent fusion
    //----------------------------------------------------------

    std::size_t i = 0;

    while (i + 1 < block.instructions.size())
    {
        if (can_fuse(
                context,
                i,
                i + 1))
        {
            merge(
                context,
                blockId,
                i,
                i + 1);

            //--------------------------------------------------
            // remove fused instruction
            //--------------------------------------------------

            block.instructions.erase(
                block.instructions.begin() + i + 1);

            ++context.result.instructionCount;

            continue;
        }

        ++i;
    }
}

//==============================================================

bool InstructionFusion::can_fuse(
    const ExecutionContext&,
    std::size_t,
    std::size_t) const
{
    //----------------------------------------------------------
    // Phase 8
    //
    // Conservative implementation.
    //
    // Full opcode analysis will arrive in Phase 9.
    //----------------------------------------------------------

    return true;
}

//==============================================================

void InstructionFusion::merge(
    ExecutionContext& context,
    std::size_t,
    std::size_t,
    std::size_t)
{
    //----------------------------------------------------------
    // Runtime statistics
    //----------------------------------------------------------

    ++context.executedInstructions;

    ++context.result.floatingPointOperations;

    context.executionEfficiency =
        context.executionEfficiency * 1.01f;

    if (context.executionEfficiency > 1.0f)
        context.executionEfficiency = 1.0f;
}

}
}