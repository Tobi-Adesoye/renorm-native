#pragma once

#include "renorm/runtime/frontier.h"
#include "renorm/runtime/region_frontier.h"
#include "renorm/runtime/runtime_result.h"
#include "renorm/runtime/compiled_graph.h"
#include "renorm/runtime/region_schedule_item.h"

#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

// Forward declaration to break the circular include loop with execution_policy.h
struct ExecutionPolicy;

//--------------------------------------------------------------
// PredictionRuntimeState
//
// Holds cached prediction metrics to avoid recomputing every frame.
//--------------------------------------------------------------
struct PredictionRuntimeState
{
    float confidence = 1.0f;
    std::size_t estimatedNextFrontierSize = 0;
    double estimatedSearchCost = 0.0;
};

//==============================================================
// ExecutionContext
//
// Central runtime state.
//
// Compile-time objects remain immutable.
//
// Every runtime system receives exactly one ExecutionContext.
//
// Graph
//   ↓
// Region Scheduler
//   ↓
// Block Executor
//   ↓
// Reasoning Engine
//   ↓
// Runtime Result
//==============================================================

struct ExecutionContext
{
    //----------------------------------------------------------
    // Immutable compiled graph
    //----------------------------------------------------------

    const CompiledGraph* graph = nullptr;

    //----------------------------------------------------------
    // Runtime frontier
    //----------------------------------------------------------

    Frontier frontier;

    //----------------------------------------------------------
    // Region frontier
    //----------------------------------------------------------

    RegionFrontier regionFrontier;

    //----------------------------------------------------------
    // Current execution policy
    //----------------------------------------------------------

    ExecutionPolicy* policy = nullptr;

    //----------------------------------------------------------
    // Runtime result
    //----------------------------------------------------------

    RuntimeResult result;

    //----------------------------------------------------------
    // Prediction state cache
    //----------------------------------------------------------

    PredictionRuntimeState prediction;

    //----------------------------------------------------------
    // Region execution schedule
    //----------------------------------------------------------

    std::vector<RegionScheduleItem> schedule;

    //----------------------------------------------------------
    // Current execution state
    //----------------------------------------------------------

    std::size_t currentRegion = 0;

    std::size_t currentBlock = 0;

    std::size_t currentInstruction = 0;

    //----------------------------------------------------------
    // Iteration state
    //----------------------------------------------------------

    std::size_t iteration = 0;

    std::size_t executedRegions = 0;

    std::size_t executedBlocks = 0;

    std::size_t executedInstructions = 0;

    //----------------------------------------------------------
    // Parallel execution statistics
    //----------------------------------------------------------

    std::size_t workerCount = 1;

    std::size_t activeWorkers = 1;

    std::size_t idleWorkers = 0;

    //----------------------------------------------------------
    // Adaptive runtime
    //----------------------------------------------------------

    float pruningRatio = 0.0f;

    float workloadBalance = 1.0f;

    float executionEfficiency = 1.0f;

    //----------------------------------------------------------
    // Runtime optimizer state
    //----------------------------------------------------------

    bool optimizerDirty = true;

    bool cacheValid = false;

    //----------------------------------------------------------
    // Runtime flags
    //----------------------------------------------------------

    bool terminate = false;

    bool goalReached = false;

    bool frontierChanged = false;

    bool regionsRescheduled = false;

    //----------------------------------------------------------
    // Reset runtime state
    //----------------------------------------------------------

    inline void clear_runtime()
    {
        frontier = Frontier();

        regionFrontier.clear();

        schedule.clear();

        currentRegion = 0;
        currentBlock = 0;
        currentInstruction = 0;

        iteration = 0;

        executedRegions = 0;
        executedBlocks = 0;
        executedInstructions = 0;

        workerCount = 1;
        activeWorkers = 1;
        idleWorkers = 0;

        pruningRatio = 0.0f;
        workloadBalance = 1.0f;
        executionEfficiency = 1.0f;

        optimizerDirty = true;
        cacheValid = false;

        terminate = false;
        goalReached = false;
        frontierChanged = false;
        regionsRescheduled = false;

        prediction = PredictionRuntimeState();
        result = RuntimeResult();
    }

    //----------------------------------------------------------
    // Attach compiled graph
    //----------------------------------------------------------

    inline void attach(const CompiledGraph& compiled)
    {
        graph = &compiled;

        clear_runtime();
    }

    //----------------------------------------------------------
    // Convenience
    //----------------------------------------------------------

    inline bool finished() const
    {
        return terminate || goalReached;
    }

    inline std::size_t region_count() const
    {
        return graph ? graph->compiledRegions.size() : 0;
    }

    inline std::size_t block_count() const
    {
        return graph ? graph->compiledBlocks.size() : 0;
    }

    inline std::size_t node_count() const
    {
        return graph ? graph->nodeCount : 0;
    }
};

} // namespace runtime
} // namespace renorm