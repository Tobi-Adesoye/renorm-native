#include "renorm/runtime/execution_profiler.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void ExecutionProfiler::initialize(
    const ExecutionContext& context)
{
    if (!context.graph)
        return;

    blockProfiles.clear();
    regionProfiles.clear();

    blockProfiles.resize(
        context.graph->compiledBlocks.size());

    regionProfiles.resize(
        context.graph->compiledRegions.size());

    runtime = RuntimeProfile();
}

//==============================================================

void ExecutionProfiler::begin_iteration(
    ExecutionContext&)
{
    ++runtime.iterations;
}

//==============================================================

void ExecutionProfiler::end_iteration(
    ExecutionContext& context)
{
    //----------------------------------------------------------
    // Runtime
    //----------------------------------------------------------

    runtime.totalNodesVisited +=
        context.frontier.nodesVisited;

    runtime.totalBlocksExecuted +=
        context.executedBlocks;

    runtime.totalInstructionsExecuted +=
        context.executedInstructions;

    runtime.totalFloatingPointOperations +=
        context.frontier.floatingPointOperations;

    runtime.totalExecutionTimeMs +=
        context.result.executionTimeMs;

    runtime.averageExecutionEfficiency =
        context.executionEfficiency;

    runtime.averagePruningRatio =
        context.pruningRatio;

    //----------------------------------------------------------
    // Block Profiles
    //----------------------------------------------------------

    for (std::size_t i = 0;
         i < context.graph->compiledBlocks.size();
         ++i)
    {
        auto& profile =
            blockProfiles[i];

        const auto& block =
            context.graph->compiledBlocks[i];

        ++profile.executions;

        profile.instructions +=
            block.instructionCount;

        profile.floatingPointOperations +=
            block.instructionCount;

        profile.executionTimeMs +=
            context.result.executionTimeMs;

        profile.averageDensity =
            block.density;
    }

    //----------------------------------------------------------
    // Region Profiles
    //----------------------------------------------------------

    for (std::size_t i = 0;
         i < context.graph->compiledRegions.size();
         ++i)
    {
        auto& profile =
            regionProfiles[i];

        const auto& region =
            context.graph->compiledRegions[i];

        ++profile.executions;

        profile.blocksExecuted +=
            region.blocks.size();

        profile.executionTimeMs +=
            context.result.executionTimeMs;

        profile.averageConfidence =
            region.confidence;

        profile.averageHeuristic =
            region.heuristic;
    }
}

//==============================================================

const RuntimeProfile&
ExecutionProfiler::runtime_profile() const
{
    return runtime;
}

//==============================================================

const BlockProfile&
ExecutionProfiler::block_profile(
    std::size_t blockId) const
{
    return blockProfiles[blockId];
}

//==============================================================

const RegionProfile&
ExecutionProfiler::region_profile(
    std::size_t regionId) const
{
    return regionProfiles[regionId];
}

} // namespace runtime
} // namespace renorm