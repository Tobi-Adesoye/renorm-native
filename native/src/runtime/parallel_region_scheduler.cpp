#include "renorm/runtime/parallel_region_scheduler.h"

#include <future>

namespace renorm
{
namespace runtime
{

//==============================================================

ParallelRegionScheduler::ParallelRegionScheduler(
    std::size_t threadCount)
    :
    threadPool_(threadCount)
{
}

//==============================================================

void ParallelRegionScheduler::schedule(
    VectorFrontier& frontier,
    const CompiledGraph& graph,
    const std::vector<std::size_t>& activeRegions)
{
    //----------------------------------------------------------
    // Nothing to execute
    //----------------------------------------------------------

    if (activeRegions.empty())
        return;

    //----------------------------------------------------------
    // Launch one task per active region
    //----------------------------------------------------------

    std::vector<
        std::future<void>> jobs;

    jobs.reserve(
        activeRegions.size());

    for (std::size_t regionId :
         activeRegions)
    {
        if (regionId >=
            graph.compiledBlocks.size())
        {
            continue;
        }

        jobs.emplace_back(

            threadPool_.submit(

                [this,
                 &frontier,
                 &graph,
                 regionId]()
                {
                    blockExecutor_.execute(

                        frontier,

                        graph.compiledBlocks[
                            regionId]);
                }));
    }

    //----------------------------------------------------------
    // Synchronize
    //----------------------------------------------------------

    for (auto& job :
         jobs)
    {
        job.get();
    }

    //----------------------------------------------------------
    // Advance frontier
    //----------------------------------------------------------

    frontier.swap_buffers();
}

} // namespace runtime
} // namespace renorm