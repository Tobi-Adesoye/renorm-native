#include "renorm/runtime/frontier_executor_gemm.h"

#include "renorm/runtime/compiled_block.h"
#include "renorm/runtime/reasoning/reasoning_engine.h"

#include <algorithm>
#include <vector>

namespace renorm
{
namespace runtime
{

void FrontierExecutorGEMM::execute(
    Frontier& frontier,
    const CompiledGraph& compiled)
{
    while (!frontier.goalReached &&
           !frontier.activeBlocks.empty())
    {
        execute_iteration(
            frontier,
            compiled);
    }

    if (frontier.goalReached &&
        frontier.goalNode != INVALID_NODE)
    {
        frontier.pathLength =
            frontier.depth[
                frontier.goalNode];
    }

    RuntimeResult dummy;

    ReasoningEngine reasoning;

    reasoning.execute(
        frontier,
        compiled,
        dummy);
}

void FrontierExecutorGEMM::execute_iteration(
    Frontier& frontier,
    const CompiledGraph& compiled)
{
    //----------------------------------------------------------
    // Reset next iteration
    //----------------------------------------------------------

    std::fill(
        frontier.next.begin(),
        frontier.next.end(),
        0.0f);

    frontier.nextActive.clear();
    frontier.nextActiveBlocks.clear();

    frontier.activeNodeCount = 0;

    //----------------------------------------------------------
    // Prevent duplicate block activation
    //----------------------------------------------------------

    std::vector<uint8_t> scheduled(
        compiled.compiledBlocks.size(),
        0);

    //----------------------------------------------------------
    // Execute active blocks
    //----------------------------------------------------------

    for (std::size_t blockId :
         frontier.activeBlocks)
    {
        const CompiledBlock& block =
            compiled.compiledBlocks[blockId];

        frontier.blocksExecuted++;

        //------------------------------------------------------
        // Execute compiled instruction stream
        //------------------------------------------------------

        for (const ExecutionInstruction& instruction :
             block.program)
        {
            const NodeId u =
                instruction.from;

            if (frontier.current[u] == 0.0f)
                continue;

            frontier.floatingPointOperations++;

            const NodeId v =
                instruction.to;

            if (frontier.visited[v])
                continue;

            frontier.visited[v] = 1;

            frontier.parent[v] = u;

            frontier.distance[v] =
                frontier.distance[u] +
                instruction.weight;

            frontier.depth[v] =
                frontier.depth[u] + 1;

            frontier.next[v] = 1.0f;

            frontier.nextActive.push_back(v);

            frontier.nodesVisited++;

            frontier.activeNodeCount++;

            //--------------------------------------------------
            // Activate destination block
            //--------------------------------------------------

            const std::size_t destinationBlock =
                compiled.nodeToCompiledBlock[v];

            if (destinationBlock !=
                    static_cast<std::size_t>(-1) &&
                !scheduled[destinationBlock])
            {
                scheduled[destinationBlock] = 1;

                frontier.nextActiveBlocks.push_back(
                    destinationBlock);
            }

            //--------------------------------------------------
            // Goal
            //--------------------------------------------------

            if (v == frontier.goalNode)
            {
                frontier.goalReached = true;

                frontier.pathLength = 0;

                NodeId current = v;

                while (current != frontier.startNode &&
                       current != INVALID_NODE)
                {
                    current =
                        frontier.parent[current];

                    frontier.pathLength++;
                }
            }
        }
    }

    //----------------------------------------------------------
    // Advance
    //----------------------------------------------------------

    frontier.swap_buffers();

    frontier.iteration++;
}

} // namespace runtime
} // namespace renorm