#include "renorm/runtime/simd_block_executor.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void SIMDBlockExecutor::execute(
    VectorFrontier& frontier,
    Frontier& scalarFrontier,
    const CompiledGraph& compiled,
    std::size_t blockId)
{
    if (blockId >= compiled.compiledBlocks.size())
        return;

    const CompiledBlock& block =
        compiled.compiledBlocks[blockId];

    scalarFrontier.blocksExecuted++;

    execute_instruction_stream(
        frontier,
        scalarFrontier,
        block,
        compiled);
}

//==============================================================

void SIMDBlockExecutor::execute_instruction_stream(
    VectorFrontier& frontier,
    Frontier& scalarFrontier,
    const CompiledBlock& block,
    const CompiledGraph& compiled)
{
    float* current =
        frontier.current();

    float* next =
        frontier.next();

    //----------------------------------------------------------

    for (const ExecutionInstruction& instruction :
         block.program)
    {
        const NodeId u =
            instruction.from;

        if (current[u] == 0.0f)
            continue;

        scalarFrontier.floatingPointOperations++;

        const NodeId v =
            instruction.to;

        if (scalarFrontier.visited[v])
            continue;

        //------------------------------------------------------
        // Activate
        //------------------------------------------------------

        scalarFrontier.visited[v] = 1;

        scalarFrontier.parent[v] = u;

        scalarFrontier.distance[v] =
            scalarFrontier.distance[u] +
            instruction.weight;

        scalarFrontier.depth[v] =
            scalarFrontier.depth[u] + 1;

        next[v] = 1.0f;

        scalarFrontier.nextActive.push_back(v);

        scalarFrontier.nodesVisited++;

        scalarFrontier.activeNodeCount++;

        //------------------------------------------------------
        // Destination block activation
        //------------------------------------------------------

        std::size_t destinationBlock =
            compiled.nodeToCompiledBlock[v];

        if (destinationBlock !=
            static_cast<std::size_t>(-1))
        {
            scalarFrontier.nextActiveBlocks.push_back(
                destinationBlock);
        }

        //------------------------------------------------------
        // Goal
        //------------------------------------------------------

        if (v == scalarFrontier.goalNode)
        {
            scalarFrontier.goalReached = true;

            scalarFrontier.pathLength = 0;

            NodeId currentNode = v;

            while (currentNode !=
                       scalarFrontier.startNode &&
                   currentNode !=
                       INVALID_NODE)
            {
                currentNode =
                    scalarFrontier.parent[currentNode];

                scalarFrontier.pathLength++;
            }
        }
    }
}

} // namespace runtime
} // namespace renorm