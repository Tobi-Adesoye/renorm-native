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
// BlockProfile
//
//==============================================================

struct BlockProfile
{
    std::size_t executions = 0;

    std::size_t instructions = 0;

    std::size_t floatingPointOperations = 0;

    double executionTimeMs = 0.0;

    float averageDensity = 0.0f;
};

//==============================================================
//
// RegionProfile
//
//==============================================================

struct RegionProfile
{
    std::size_t executions = 0;

    std::size_t blocksExecuted = 0;

    double executionTimeMs = 0.0;

    float averageConfidence = 0.0f;

    float averageHeuristic = 0.0f;
};

//==============================================================
//
// RuntimeProfile
//
//==============================================================

struct RuntimeProfile
{
    std::size_t iterations = 0;

    std::size_t totalNodesVisited = 0;

    std::size_t totalBlocksExecuted = 0;

    std::size_t totalInstructionsExecuted = 0;

    std::size_t totalFloatingPointOperations = 0;

    double totalExecutionTimeMs = 0.0;

    float averageExecutionEfficiency = 0.0f;

    float averagePruningRatio = 0.0f;
};

//==============================================================
//
// ExecutionProfiler
//
// Collects runtime execution statistics.
//
//==============================================================

class ExecutionProfiler
{
public:

    ExecutionProfiler() = default;

    //----------------------------------------------------------

    void initialize(
        const ExecutionContext& context);

    //----------------------------------------------------------

    void begin_iteration(
        ExecutionContext& context);

    //----------------------------------------------------------

    void end_iteration(
        ExecutionContext& context);

    //----------------------------------------------------------

    const RuntimeProfile&
    runtime_profile() const;

    //----------------------------------------------------------

    const BlockProfile&
    block_profile(
        std::size_t blockId) const;

    //----------------------------------------------------------

    const RegionProfile&
    region_profile(
        std::size_t regionId) const;

private:

    RuntimeProfile runtime;

    std::vector<BlockProfile> blockProfiles;

    std::vector<RegionProfile> regionProfiles;
};

}
}