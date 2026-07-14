#include "renorm/runtime/parallel_statistics.h"

namespace renorm
{
namespace runtime
{

//==============================================================

ParallelStatistics::ParallelStatistics(
    std::size_t workerCount)
    :
    workers_(workerCount)
{
}

//==============================================================

void
ParallelStatistics::reset()
{
    for (auto& worker : workers_)
    {
        worker.instructionsExecuted.store(0);

        worker.blocksExecuted.store(0);

        worker.regionsExecuted.store(0);

        worker.cacheHits.store(0);

        worker.cacheMisses.store(0);

        worker.queueSteals.store(0);

        worker.idleCycles.store(0);

        worker.synchronizationCount.store(0);
    }
}

//==============================================================

ParallelStatistics::WorkerStatistics&
ParallelStatistics::worker(
    std::size_t id)
{
    return workers_[id];
}

const ParallelStatistics::WorkerStatistics&
ParallelStatistics::worker(
    std::size_t id) const
{
    return workers_[id];
}

//==============================================================

std::size_t
ParallelStatistics::totalInstructions() const
{
    std::size_t total = 0;

    for (const auto& worker : workers_)
        total += worker.instructionsExecuted.load();

    return total;
}

//==============================================================

std::size_t
ParallelStatistics::totalBlocks() const
{
    std::size_t total = 0;

    for (const auto& worker : workers_)
        total += worker.blocksExecuted.load();

    return total;
}

//==============================================================

std::size_t
ParallelStatistics::totalRegions() const
{
    std::size_t total = 0;

    for (const auto& worker : workers_)
        total += worker.regionsExecuted.load();

    return total;
}

//==============================================================

std::size_t
ParallelStatistics::totalCacheHits() const
{
    std::size_t total = 0;

    for (const auto& worker : workers_)
        total += worker.cacheHits.load();

    return total;
}

//==============================================================

std::size_t
ParallelStatistics::totalCacheMisses() const
{
    std::size_t total = 0;

    for (const auto& worker : workers_)
        total += worker.cacheMisses.load();

    return total;
}

//==============================================================

std::size_t
ParallelStatistics::totalSteals() const
{
    std::size_t total = 0;

    for (const auto& worker : workers_)
        total += worker.queueSteals.load();

    return total;
}

//==============================================================

std::size_t
ParallelStatistics::totalIdleCycles() const
{
    std::size_t total = 0;

    for (const auto& worker : workers_)
        total += worker.idleCycles.load();

    return total;
}

//==============================================================

std::size_t
ParallelStatistics::totalSynchronizations() const
{
    std::size_t total = 0;

    for (const auto& worker : workers_)
        total += worker.synchronizationCount.load();

    return total;
}

//==============================================================

std::size_t
ParallelStatistics::workerCount() const noexcept
{
    return workers_.size();
}

} // namespace runtime
} // namespace renorm