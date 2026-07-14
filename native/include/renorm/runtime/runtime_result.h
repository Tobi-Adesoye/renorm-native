#pragma once

#include <cstddef>

#include "renorm/search/search_result.h"

namespace renorm
{
namespace runtime
{

struct RuntimeResult
{
    //----------------------------------------------------------
    // Search
    //----------------------------------------------------------

    SearchResult search;

    //----------------------------------------------------------
    // Timing
    //----------------------------------------------------------

    double compileTimeMs = 0.0;
    double executionTimeMs = 0.0;
    double totalTimeMs = 0.0;

    //----------------------------------------------------------
    // Runtime
    //----------------------------------------------------------

    std::size_t nodesVisited = 0;
    std::size_t pathLength = 0;

    std::size_t blocksExecuted = 0;
    std::size_t floatingPointOperations = 0;

    //----------------------------------------------------------
    // Prediction
    //----------------------------------------------------------

    std::size_t predictedVisited = 0;

    float predictionAccuracy = 0.0f;

    float predictedFrontierSize = 0.0f;

    float predictedGoalDistance = 0.0f;

    float predictedExpansionCost = 0.0f;

    float predictedBranchExpansion = 0.0f;

    float predictionConfidence = 0.0f;

    //----------------------------------------------------------
    // Structural graph metrics
    //----------------------------------------------------------

    float frontierDensity = 0.0f;

    float averageBranchingFactor = 0.0f;

    std::size_t corridorNodes = 0;

    std::size_t hubNodes = 0;

    std::size_t bottleneckNodes = 0;

    std::size_t repeatedTraversalCount = 0;

    //----------------------------------------------------------
    // Reasoning
    //----------------------------------------------------------

    float confidence = 0.0f;

    float decisionCost = 0.0f;

    float expansionReduction = 0.0f;

    float executionPriority = 0.0f;

    float expansionPolicy = 0.0f;

    float frontierQuality = 0.0f;

    float graphComplexity = 0.0f;

    float heuristicScore = 0.0f;
};

}
}