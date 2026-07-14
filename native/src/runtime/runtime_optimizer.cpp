#include "renorm/runtime/runtime_optimizer.h"

#include "renorm/runtime/execution_context.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void RuntimeOptimizer::optimize(
    ExecutionContext& context) const
{
    optimize_blocks(context);

    optimize_regions(context);

    optimize_branches(context);

    optimize_simd(context);

    optimize_cache(context);

    optimize_memory(context);

    optimize_parallelism(context);

    //----------------------------------------------------------
    // Update execution efficiency
    //----------------------------------------------------------

    context.executionEfficiency =
        compute_execution_efficiency(
            context);

    context.cacheEfficiency =
        compute_cache_efficiency(
            context);

    context.parallelEfficiency =
        compute_parallel_efficiency(
            context);
}

//==============================================================

void RuntimeOptimizer::optimize_blocks(
    ExecutionContext& context) const
{
    //----------------------------------------------------------
    // Dense blocks receive higher priority
    //----------------------------------------------------------

    if (context.frontier.frontierDensity > 0.70f)
    {
        context.blockFusionEnabled = true;
    }
}

//==============================================================

void RuntimeOptimizer::optimize_regions(
    ExecutionContext& context) const
{
    if (context.frontier.averageBranchingFactor >
        4.0f)
    {
        context.regionFusionEnabled = true;
    }
}

//==============================================================

void RuntimeOptimizer::optimize_branches(
    ExecutionContext& context) const
{
    if (context.frontier.predictionConfidence >
        0.80f)
    {
        context.branchPruningEnabled = true;
    }
}

//==============================================================

void RuntimeOptimizer::optimize_simd(
    ExecutionContext& context) const
{
    if (context.simdAvailable)
    {
        context.simdExecutionEnabled = true;
    }
}

//==============================================================

void RuntimeOptimizer::optimize_cache(
    ExecutionContext& context) const
{
    if (context.frontier.frontierQuality >
        0.60f)
    {
        context.cacheOptimizationEnabled = true;
    }
}

//==============================================================

void RuntimeOptimizer::optimize_memory(
    ExecutionContext& context) const
{
    if (context.memoryPressure >
        0.75f)
    {
        context.memoryCompactionEnabled = true;
    }
}

//==============================================================

void RuntimeOptimizer::optimize_parallelism(
    ExecutionContext& context) const
{
    if (context.parallelEfficiency <
        0.60f)
    {
        context.dynamicLoadBalancing = true;
    }
}

//==============================================================

float RuntimeOptimizer::compute_execution_efficiency(
    const ExecutionContext& context) const
{
    float score = 0.0f;

    score += context.frontier.frontierQuality;

    score += context.frontier.predictionConfidence;

    score += context.frontier.heuristicScore;

    score /= 3.0f;

    return std::clamp(
        score,
        0.0f,
        1.0f);
}

//==============================================================

float RuntimeOptimizer::compute_cache_efficiency(
    const ExecutionContext& context) const
{
    float score = 1.0f;

    score -= context.cacheMissRate;

    return std::clamp(
        score,
        0.0f,
        1.0f);
}

//==============================================================

float RuntimeOptimizer::compute_parallel_efficiency(
    const ExecutionContext& context) const
{
    return std::clamp(
        context.workloadBalance,
        0.0f,
        1.0f);
}

} // namespace runtime
} // namespace renorm