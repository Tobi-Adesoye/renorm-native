#include "renorm/runtime/frontier_builder.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================
// Legacy API
//==============================================================

Frontier FrontierBuilder::build(
    const CompiledGraph& compiled,
    NodeId start,
    NodeId goal) const
{
    Frontier frontier;

    initialize_frontier(
        frontier,
        compiled,
        start,
        goal);

    return frontier;
}

//==============================================================
// ExecutionContext API
//==============================================================

void FrontierBuilder::build(
    ExecutionContext& context,
    NodeId start,
    NodeId goal) const
{
    if (context.graph == nullptr)
        return;

    initialize_frontier(
        context.frontier,
        *context.graph,
        start,
        goal);

    //----------------------------------------------------------
    // Region frontier
    //----------------------------------------------------------

    if (!context.graph->blockToRegion.empty())
    {
        std::size_t startBlock =
            context.graph->nodeToCompiledBlock[start];

        std::size_t startRegion =
            context.graph->blockToRegion[startBlock];

        context.regionFrontier.build(
            *context.graph,
            startRegion);

        context.currentRegion =
            startRegion;
    }

    context.currentBlock = 0;
    context.currentInstruction = 0;

    context.iteration = 0;

    context.goalReached = false;
    context.terminate = false;
}

//==============================================================
// Internal
//==============================================================

void FrontierBuilder::initialize_frontier(
    Frontier& frontier,
    const CompiledGraph& compiled,
    NodeId start,
    NodeId goal) const
{
    frontier.resize(
        compiled.nodeCount);

    frontier.startNode =
        start;

    frontier.goalNode =
        goal;

    frontier.goalReached =
        false;

    //----------------------------------------------------------
    // Activate start node
    //----------------------------------------------------------

    frontier.current[start] = 1.0f;

    frontier.active.push_back(start);

    frontier.visited[start] = 1;

    frontier.parent[start] =
        INVALID_NODE;

    frontier.distance[start] =
        0.0f;

    frontier.depth[start] =
        0;

    //----------------------------------------------------------
    // Activate owning block
    //----------------------------------------------------------

    if (!compiled.nodeToCompiledBlock.empty())
    {
        std::size_t block =
            compiled.nodeToCompiledBlock[start];

        frontier.activeBlocks.push_back(
            block);
    }

    //----------------------------------------------------------
    // Reset counters
    //----------------------------------------------------------

    frontier.iteration = 0;

    frontier.nodesVisited = 1;

    frontier.activeNodeCount = 1;

    frontier.blocksExecuted = 0;

    frontier.floatingPointOperations = 0;

    frontier.pathLength = 0;

    //----------------------------------------------------------
    // Reset reasoning metrics
    //----------------------------------------------------------

    frontier.frontierDensity = 0.0f;

    frontier.averageBranchingFactor = 0.0f;

    frontier.corridorNodes = 0;

    frontier.hubNodes = 0;

    frontier.bottleneckNodes = 0;

    frontier.repeatedTraversalCount = 0;

    frontier.predictedFrontierSize = 0.0f;

    frontier.predictedGoalDistance = 0.0f;

    frontier.predictedExpansionCost = 0.0f;

    frontier.predictedBranchExpansion = 0.0f;

    frontier.predictionConfidence = 0.0f;

    frontier.frontierQuality = 0.0f;

    frontier.graphComplexity = 0.0f;

    frontier.heuristicScore = 0.0f;

    frontier.confidence = 0.0f;

    frontier.decisionCost = 0.0f;

    frontier.expansionReduction = 0.0f;

    frontier.executionPriority = 0.0f;

    frontier.expansionPolicy = 0.0f;
}

}
}