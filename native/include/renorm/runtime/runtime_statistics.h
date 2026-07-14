#pragma once

#include "renorm/runtime/execution_context.h"

#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// RuntimeStatistics
//
// Aggregated execution statistics for the complete runtime.
//
//==============================================================

struct RuntimeStatistics
{
    //----------------------------------------------------------
    // Execution
    //----------------------------------------------------------

    std::size_t iterations = 0;

    std::size_t nodesVisited = 0;

    std::size_t blocksExecuted = 0;

    std::size_t regionsExecuted = 0;

    std::size_t instructionsExecuted = 0;

    std::size_t floatingPointOperations = 0;

    //----------------------------------------------------------
    // Timing
    //----------------------------------------------------------

    double compileTimeMs = 0.0;

    double executionTimeMs = 0.0;

    double totalTimeMs = 0.0;

    //----------------------------------------------------------
    // Adaptive Runtime
    //----------------------------------------------------------

    float executionEfficiency = 1.0f;

    float workloadBalance = 1.0f;

    float pruningRatio = 0.0f;

    //----------------------------------------------------------
    // Prediction
    //----------------------------------------------------------

    float predictionConfidence = 0.0f;

    float predictionAccuracy = 0.0f;

    float heuristicScore = 0.0f;

    //----------------------------------------------------------
    // Graph
    //----------------------------------------------------------

    float frontierDensity = 0.0f;

    float frontierQuality = 0.0f;

    float graphComplexity = 0.0f;

    //----------------------------------------------------------

    void clear();
};

//==============================================================
//
// RuntimeStatisticsCollector
//
// Copies runtime information from ExecutionContext.
//
//==============================================================

class RuntimeStatisticsCollector
{
public:

    RuntimeStatisticsCollector() = default;

    //----------------------------------------------------------

    void collect(
        const ExecutionContext& context,
        RuntimeStatistics& statistics) const;

    //----------------------------------------------------------

    void accumulate(
        const RuntimeStatistics& frame,
        RuntimeStatistics& total) const;
};

}
}