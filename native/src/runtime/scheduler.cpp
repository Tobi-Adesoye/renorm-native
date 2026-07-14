#include "renorm/worker_context.h"
#include "renorm/scheduler.h"
#include "renorm/thread_pool.h"

#include <algorithm>
#include <future>
#include <vector>

namespace renorm
{

//==============================================================
// Global Thread Pool
//
// Constructed once and reused throughout the lifetime of the
// process.
//
//==============================================================

static ThreadPool gThreadPool;

//==============================================================
// parallel_for
//==============================================================

void parallel_for(
    int begin,
    int end,
    int grainSize,
    const std::function<void(
        WorkerContext&,
        int,
        int)>& fn)
{
    //----------------------------------------------------------
    // Trivial Cases
    //----------------------------------------------------------

    if (end <= begin)
    {
        return;
    }

    const int totalWork = end - begin;

    if (totalWork <= grainSize)
    {
        WorkerContext worker;

        fn(
            worker,
            begin,
            end);

        return;
    }

    //----------------------------------------------------------
    // Worker Count
    //----------------------------------------------------------

    const int workerCount =
        static_cast<int>(gThreadPool.size());

    const int chunk =
        std::max(
            grainSize,
            (totalWork + workerCount - 1) / workerCount);

    //----------------------------------------------------------
    // Launch Tasks
    //----------------------------------------------------------

    std::vector<std::future<void>> futures;

    for (int start = begin;
         start < end;
         start += chunk)
    {
        const int stop =
            std::min(
                end,
                start + chunk);

        futures.emplace_back(

            gThreadPool.enqueue(

                [=](WorkerContext& worker)
                {
                    fn(
                        worker,
                        start,
                        stop);
                }));
    }

    //----------------------------------------------------------
    // Wait
    //----------------------------------------------------------

    for (auto& future : futures)
    {
        future.get();
    }
}

} // namespace renorm