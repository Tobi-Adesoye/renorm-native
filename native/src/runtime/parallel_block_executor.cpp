#include "renorm/runtime/parallel_block_executor.h"

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// Execute Region
//--------------------------------------------------------------

void ParallelBlockExecutor::execute(
    ExecutionContext& context)
{
    if (context.graph == nullptr)
        return;

    if (context.currentRegion >=
        context.graph->compiledRegions.size())
        return;

    const CompiledRegion& region =
        context.graph->compiledRegions[
            context.currentRegion];

    //----------------------------------------------------------
    // Execute every block
    //----------------------------------------------------------

    for (std::size_t blockId : region.blocks)
    {
        execute_block(
            context,
            blockId);
    }
}

//--------------------------------------------------------------
// Execute Block
//--------------------------------------------------------------

void ParallelBlockExecutor::execute_block(
    ExecutionContext& context,
    std::size_t blockId)
{
    const CompiledBlock& block =
        context.graph->compiledBlocks[blockId];

    context.currentBlock = blockId;

    context.result.blocksExecuted++;

    //----------------------------------------------------------
    // Execute instruction stream
    //----------------------------------------------------------

    for (const auto& instruction : block.program)
    {
        execute_instruction(
            context,
            instruction);
    }
}

//--------------------------------------------------------------
// Execute Instruction
//--------------------------------------------------------------

void ParallelBlockExecutor::execute_instruction(
    ExecutionContext& context,
    const ExecutionInstruction& instruction)
{
    Frontier& frontier =
        context.frontier;

    //----------------------------------------------------------
    // Skip inactive node
    //----------------------------------------------------------

    if (frontier.current[instruction.from] <= 0.0f)
        return;

    //----------------------------------------------------------
    // Propagate activation
    //----------------------------------------------------------

    float value =
        frontier.current[instruction.from] *
        instruction.weight;

    if (value > frontier.next[instruction.to])
    {
        frontier.next[instruction.to] = value;

        frontier.nextActive.push_back(
            instruction.to);
    }

    //----------------------------------------------------------
    // Visit bookkeeping
    //----------------------------------------------------------

    if (!frontier.visited[instruction.to])
    {
        frontier.visited[instruction.to] = 1;

        frontier.parent[instruction.to] =
            instruction.from;

        frontier.nodesVisited++;

        context.result.nodesVisited++;
    }

    //----------------------------------------------------------
    // Goal reached
    //----------------------------------------------------------

    if (instruction.to == frontier.goalNode)
    {
        frontier.goalReached = true;

        context.goalReached = true;
    }

    //----------------------------------------------------------
    // Runtime statistics
    //----------------------------------------------------------

    frontier.floatingPointOperations++;

    context.result.floatingPointOperations++;
}

}
}