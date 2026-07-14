#pragma once

#include <vector>
#include <thread>

#include "renorm/runtime/vector_frontier.h"
#include "renorm/runtime/compiled_graph.h"

#include "renorm/runtime/thread_pool.h"
#include "renorm/runtime/region_stealer.h"
#include "renorm/runtime/numa_scheduler.h"
#include "renorm/runtime/cpu_affinity_manager.h"
#include "renorm/runtime/simd_block_executor.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// ParallelFrontierExpansion
//
// High-performance multicore frontier expansion.
//
// Responsibilities:
//
//  • distribute active regions
//  • work stealing
//  • NUMA-aware scheduling
//  • CPU affinity
//  • SIMD execution
//
//==============================================================

class ParallelFrontierExpansion
{
public:

    explicit ParallelFrontierExpansion(
        std::size_t workerCount =
            std::thread::hardware_concurrency());

    //----------------------------------------------------------
    // Execute one frontier iteration
    //----------------------------------------------------------

    void execute(
        VectorFrontier& frontier,
        const CompiledGraph& graph,
        const std::vector<std::size_t>& activeRegions);

private:

    void worker_loop(
        std::size_t workerId,
        VectorFrontier& frontier,
        const CompiledGraph& graph);

private:

    std::size_t workerCount_;

    ThreadPool threadPool_;

    RegionStealer stealer_;

    NUMAScheduler numa_;

    CPUAffinityManager affinity_;

    SIMDBlockExecutor executor_;
};

} // namespace runtime
} // namespace renorm