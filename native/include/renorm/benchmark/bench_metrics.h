#pragma once

#include <string>
#include <cstddef>

namespace renorm
{
namespace benchmark
{

struct BenchMetrics
{
    //----------------------------------------------------------
    // Identification
    //----------------------------------------------------------

    std::string algorithm;

    std::string graphFamily;

    //----------------------------------------------------------
    // Graph
    //----------------------------------------------------------

    std::size_t nodes = 0;

    std::size_t edges = 0;

    //----------------------------------------------------------
    // Timing
    //----------------------------------------------------------

    double compileTimeMs = 0.0;

    double executionTimeMs = 0.0;

    double totalTimeMs = 0.0;

    //----------------------------------------------------------
    // Search
    //----------------------------------------------------------

    std::size_t visitedNodes = 0;

    std::size_t pathLength = 0;

    //----------------------------------------------------------
    // Runtime
    //----------------------------------------------------------

    std::size_t blocksExecuted = 0;

    std::size_t floatingPointOperations = 0;

    //----------------------------------------------------------
    // Throughput
    //----------------------------------------------------------

    double nodesPerSecond = 0.0;

    double edgesPerSecond = 0.0;

    double speedup = 0.0;

    //----------------------------------------------------------
    // Reduction
    //----------------------------------------------------------

    double reductionPercent = 0.0;

    //----------------------------------------------------------
    // Reasoning Engine
    //----------------------------------------------------------

    float confidence = 0.0f;

    float heuristicScore = 0.0f;

    float frontierQuality = 0.0f;

    float graphComplexity = 0.0f;

    float predictionConfidence = 0.0f;

    float decisionCost = 0.0f;

    float expansionReduction = 0.0f;

    float executionPriority = 0.0f;

    float expansionPolicy = 0.0f;
};

}
}