#include "renorm/runtime/parallel_frontier_expansion.h"

#include <future>

namespace renorm
{
namespace runtime
{

//==============================================================

ParallelFrontierExpansion::ParallelFrontierExpansion(
    std::size_t workerCount)
    :
    workerCount_(
        workerCount == 0 ? 1 : workerCount),

    threadPool_(workerCount_),

    stealer_(workerCount_)
{
    numa_.initialize(workerCount_);
}

//==============================================================

void ParallelFrontierExpansion::execute(
    VectorFrontier& frontier,
    const CompiledGraph& graph,
    const std::vector<std::size_t>& activeRegions)
{
    //----------------------------------------------------------
    // Initialize scheduler
    //----------------------------------------------------------

    stealer_.initialize(
        activeRegions);

    //----------------------------------------------------------
    // Launch workers
    //----------------------------------------------------------

    std::vector<
        std::future<void>> futures;

    futures.reserve(workerCount_);

    for (std::size_t worker = 0;
         worker < workerCount_;
         ++worker)
    {
        futures.emplace_back(

            threadPool_.submit(

                [this,
                 worker,
                 &frontier,
                 &graph]()
                {
                    worker_loop(
                        worker,
                        frontier,
                        graph);
                }));
    }

    //----------------------------------------------------------
    // Synchronize
    //----------------------------------------------------------

    for (auto& future : futures)
    {
        future.get();
    }

    //----------------------------------------------------------
    // Advance frontier
    //----------------------------------------------------------

    frontier.swap_buffers();
}

//==============================================================

void ParallelFrontierExpansion::worker_loop(
    std::size_t workerId,
    VectorFrontier& frontier,
    const CompiledGraph& graph)
{
    //----------------------------------------------------------
    // Bind worker
    //----------------------------------------------------------

    affinity_.pin_worker(workerId);

    //----------------------------------------------------------
    // Execute until queues empty
    //----------------------------------------------------------

    while (!stealer_.empty())
    {
        //------------------------------------------------------
        // Local work
        //------------------------------------------------------

        auto region =
            stealer_.pop_local(workerId);

        //------------------------------------------------------
        // Attempt steal
        //------------------------------------------------------

        if (!region.has_value())
        {
            region =
                stealer_.steal(workerId);
        }

        if (!region.has_value())
            continue;

        //------------------------------------------------------
        // Execute SIMD block
        //------------------------------------------------------

        executor_.execute(

            frontier,

            graph.compiledBlocks[
                *region]);
    }

    //----------------------------------------------------------
    // Release affinity
    //----------------------------------------------------------

    affinity_.unpin_current_thread();
}

} // namespace runtime
} // namespace renorm