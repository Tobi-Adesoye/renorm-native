#pragma once

#include "renorm/graph.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace renorm
{
namespace runtime
{

struct Frontier
{
    //----------------------------------------------------------
    // Search endpoints
    //----------------------------------------------------------

    NodeId startNode = INVALID_NODE;
    NodeId goalNode  = INVALID_NODE;

    bool goalReached = false;

    //----------------------------------------------------------
    // Node activation buffers
    //----------------------------------------------------------

    std::vector<float> current;
    std::vector<float> next;

    //----------------------------------------------------------
    // Node state
    //----------------------------------------------------------

    std::vector<uint8_t> visited;

    std::vector<NodeId> parent;

    std::vector<float> distance;

    std::vector<std::size_t> depth;

    //----------------------------------------------------------
    // Active nodes
    //----------------------------------------------------------

    std::vector<NodeId> active;
    std::vector<NodeId> nextActive;

    //----------------------------------------------------------
    // Phase B2
    //
    // Active execution blocks
    //----------------------------------------------------------

    std::vector<std::size_t> activeBlocks;
    std::vector<std::size_t> nextActiveBlocks;

    //----------------------------------------------------------
    // Runtime counters
    //----------------------------------------------------------

    std::size_t iteration = 0;

    std::size_t nodesVisited = 0;

    std::size_t activeNodeCount = 0;

    std::size_t blocksExecuted = 0;

    std::size_t floatingPointOperations = 0;

    std::size_t pathLength = 0;

    //----------------------------------------------------------
    // Reasoning metrics
    //----------------------------------------------------------

    float frontierDensity = 0.0f;
    float averageBranchingFactor = 0.0f;

    std::size_t corridorNodes = 0;
    std::size_t hubNodes = 0;
    std::size_t bottleneckNodes = 0;
    std::size_t repeatedTraversalCount = 0;

    float predictedFrontierSize = 0.0f;
    float predictedGoalDistance = 0.0f;
    float predictedExpansionCost = 0.0f;
    float predictedBranchExpansion = 0.0f;
    float predictionConfidence = 0.0f;

    float frontierQuality = 0.0f;
    float graphComplexity = 0.0f;
    float heuristicScore = 0.0f;

    float confidence = 0.0f;
    float decisionCost = 0.0f;
    float expansionReduction = 0.0f;
    float executionPriority = 0.0f;
    float expansionPolicy = 0.0f;

    //----------------------------------------------------------
    // Resize
    //----------------------------------------------------------

    inline void resize(std::size_t nodeCount)
    {
        current.assign(nodeCount, 0.0f);
        next.assign(nodeCount, 0.0f);

        visited.assign(nodeCount, 0);

        parent.assign(nodeCount, INVALID_NODE);

        distance.assign(nodeCount, 0.0f);

        depth.assign(nodeCount, 0);

        active.clear();
        nextActive.clear();

        activeBlocks.clear();
        nextActiveBlocks.clear();
    }

    //----------------------------------------------------------
    // Advance one iteration
    //----------------------------------------------------------

    inline void swap_buffers()
    {
        current.swap(next);

        active.swap(nextActive);

        activeBlocks.swap(nextActiveBlocks);

        nextActive.clear();

        nextActiveBlocks.clear();
    }
};

} // namespace runtime
} // namespace renorm