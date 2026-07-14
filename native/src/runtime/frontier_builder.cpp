#include "renorm/runtime/frontier_builder.h"

namespace renorm
{
namespace runtime
{

Frontier FrontierBuilder::build(
    const CompiledGraph& compiled,
    NodeId start,
    NodeId goal) const
{
    Frontier frontier;

    //----------------------------------------------------------
    // Initialize search endpoints
    //----------------------------------------------------------

    frontier.startNode = start;
    frontier.goalNode  = goal;

    frontier.goalReached = false;

    //----------------------------------------------------------
    // Allocate runtime buffers
    //----------------------------------------------------------

    frontier.resize(compiled.nodeCount);

    //----------------------------------------------------------
    // Activate starting node
    //----------------------------------------------------------

    frontier.current[start] = 1.0f;

    frontier.visited[start] = 1;

    frontier.parent[start] = INVALID_NODE;

    frontier.distance[start] = 0.0f;

    frontier.depth[start] = 0;

    frontier.active.push_back(start);

    //----------------------------------------------------------
    // Activate first compiled block
    //----------------------------------------------------------

    if (!compiled.nodeToCompiledBlock.empty())
    {
        const std::size_t blockId =
            compiled.nodeToCompiledBlock[start];

        if (blockId != static_cast<std::size_t>(-1))
        {
            frontier.activeBlocks.push_back(blockId);
        }
    }

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    frontier.iteration = 0;

    frontier.nodesVisited = 1;

    frontier.activeNodeCount = 1;

    frontier.blocksExecuted = 0;

    frontier.floatingPointOperations = 0;

    frontier.pathLength = 0;

    return frontier;
}

} // namespace runtime
} // namespace renorm