#include "renorm/runtime/parallel_runtime.h"

#include "renorm/runtime/parallel_block_executor.h"

#include <thread>
#include <algorithm>

namespace renorm
{
namespace runtime
{

ParallelRuntime::ParallelRuntime()
{
    workerCount =
        std::max<std::size_t>(
            1,
            std::thread::hardware_concurrency());
}

ParallelRuntime::ParallelRuntime(
    std::size_t workersRequested)
{
    workerCount =
        std::max<std::size_t>(
            1,
            workersRequested);
}

ParallelRuntime::~ParallelRuntime()
{
    shutdown();
}

//==============================================================

void ParallelRuntime::initialize()
{
    shutdown();

    workers.reserve(workerCount);

    for (std::size_t i = 0;
         i < workerCount;
         ++i)
    {
        auto worker =
            std::make_unique<WorkerThread>();

        worker->start();

        workers.push_back(
            std::move(worker));
    }
}

//==============================================================

void ParallelRuntime::shutdown()
{
    for (auto& worker : workers)
    {
        worker->stop();
    }

    workers.clear();
}

//==============================================================

void ParallelRuntime::execute(
    ExecutionContext& context)
{
    dispatch_regions(context);

    synchronize();

    context.activeWorkers =
        workerCount;

    context.idleWorkers = 0;
}

//==============================================================

void ParallelRuntime::dispatch_regions(
    ExecutionContext& context)
{
    if (workers.empty())
        initialize();

    ParallelBlockExecutor executor;

    const std::size_t regionCount =
        context.schedule.size();

    if (regionCount == 0)
        return;

    for (std::size_t i = 0;
         i < regionCount;
         ++i)
    {
        WorkerThread& worker =
            *workers[
                i % workers.size()];

        worker.execute(
            [&context, &executor]()
            {
                executor.execute(context);
            });
    }
}

//==============================================================

void ParallelRuntime::synchronize()
{
    for (auto& worker : workers)
    {
        worker->wait();
    }
}

//==============================================================

std::size_t ParallelRuntime::worker_count() const
{
    return workerCount;
}

//==============================================================

void ParallelRuntime::resize(
    std::size_t workersRequested)
{
    shutdown();

    workerCount =
        std::max<std::size_t>(
            1,
            workersRequested);

    initialize();
}

} // namespace runtime
} // namespace renorm