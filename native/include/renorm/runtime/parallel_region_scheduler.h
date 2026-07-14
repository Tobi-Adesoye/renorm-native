#pragma once

#include <vector>
#include <cstddef>

#include "renorm/runtime/thread_pool.h"
#include "renorm/runtime/vector_frontier.h"
#include "renorm/runtime/compiled_graph.h"
#include "renorm/runtime/simd_block_executor.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// ParallelRegionScheduler
//
// Executes active compiled regions in parallel.
//
// Current strategy:
//
//  • Static scheduling
//  • ThreadPool dispatch
//
// Future:
//
//  • Work stealing
//  • NUMA scheduling
//  • CPU affinity
//
//==============================================================

class ParallelRegionScheduler
{
public:

    explicit ParallelRegionScheduler(
        std::size_t threadCount =
            std::thread::hardware_concurrency());

    //----------------------------------------------------------
    // Execute active regions
    //----------------------------------------------------------

    void schedule(
        VectorFrontier& frontier,
        const CompiledGraph& graph,
        const std::vector<std::size_t>& activeRegions);

private:

    ThreadPool threadPool_;

    SIMDBlockExecutor blockExecutor_;
};

} // namespace runtime
} // namespace renorm