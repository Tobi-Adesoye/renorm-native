#include "renorm/runtime/runtime_statistics.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void RuntimeStatistics::clear()
{
    *this = RuntimeStatistics();
}

//==============================================================

void RuntimeStatisticsCollector::collect(
    const ExecutionContext& context,
    RuntimeStatistics& statistics) const
{
    statistics.iterations =
        context.iteration;

    statistics.nodesVisited =
        context.frontier.nodesVisited;

    statistics.blocksExecuted =
        context.executedBlocks;

    statistics.regionsExecuted =
        context.executedRegions;

    statistics.instructionsExecuted =
        context.executedInstructions;

    statistics.floatingPointOperations =
        context.frontier.floatingPointOperations;

    //----------------------------------------------------------

    statistics.compileTimeMs =
        context.result.compileTimeMs;

    statistics.executionTimeMs =
        context.result.executionTimeMs;

    statistics.totalTimeMs =
        context.result.totalTimeMs;

    //----------------------------------------------------------

    statistics.executionEfficiency =
        context.executionEfficiency;

    statistics.workloadBalance =
        context.workloadBalance;

    statistics.pruningRatio =
        context.pruningRatio;

    //----------------------------------------------------------

    statistics.predictionConfidence =
        context.frontier.predictionConfidence;

    statistics.predictionAccuracy =
        context.result.predictionAccuracy;

    statistics.heuristicScore =
        context.frontier.heuristicScore;

    //----------------------------------------------------------

    statistics.frontierDensity =
        context.frontier.frontierDensity;

    statistics.frontierQuality =
        context.frontier.frontierQuality;

    statistics.graphComplexity =
        context.frontier.graphComplexity;
}

//==============================================================

void RuntimeStatisticsCollector::accumulate(
    const RuntimeStatistics& frame,
    RuntimeStatistics& total) const
{
    total.iterations +=
        frame.iterations;

    total.nodesVisited +=
        frame.nodesVisited;

    total.blocksExecuted +=
        frame.blocksExecuted;

    total.regionsExecuted +=
        frame.regionsExecuted;

    total.instructionsExecuted +=
        frame.instructionsExecuted;

    total.floatingPointOperations +=
        frame.floatingPointOperations;

    //----------------------------------------------------------

    total.compileTimeMs +=
        frame.compileTimeMs;

    total.executionTimeMs +=
        frame.executionTimeMs;

    total.totalTimeMs +=
        frame.totalTimeMs;

    //----------------------------------------------------------

    total.executionEfficiency =
        frame.executionEfficiency;

    total.workloadBalance =
        frame.workloadBalance;

    total.pruningRatio =
        frame.pruningRatio;

    //----------------------------------------------------------

    total.predictionConfidence =
        frame.predictionConfidence;

    total.predictionAccuracy =
        frame.predictionAccuracy;

    total.heuristicScore =
        frame.heuristicScore;

    //----------------------------------------------------------

    total.frontierDensity =
        frame.frontierDensity;

    total.frontierQuality =
        frame.frontierQuality;

    total.graphComplexity =
        frame.graphComplexity;
}

} // namespace runtime
} // namespace renorm