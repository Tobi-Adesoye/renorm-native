#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// ParallelStatistics
//
// Per-worker runtime statistics.
//
// Designed to avoid false sharing by aligning each worker's
// counters to a cache line.
//
//==============================================================

class ParallelStatistics
{
public:

    struct alignas(64) WorkerStatistics
    {
        std::atomic<std::size_t> instructionsExecuted{0};

        std::atomic<std::size_t> blocksExecuted{0};

        std::atomic<std::size_t> regionsExecuted{0};

        std::atomic<std::size_t> cacheHits{0};

        std::atomic<std::size_t> cacheMisses{0};

        std::atomic<std::size_t> queueSteals{0};

        std::atomic<std::size_t> idleCycles{0};

        std::atomic<std::size_t> synchronizationCount{0};

        char padding[
            64 -
            (8 * sizeof(std::atomic<std::size_t>)) > 0
            ? 64 - (8 * sizeof(std::atomic<std::size_t>))
            : 1];
    };

public:

    explicit ParallelStatistics(
        std::size_t workerCount);

    //----------------------------------------------------------
    // Reset
    //----------------------------------------------------------

    void reset();

    //----------------------------------------------------------
    // Worker access
    //----------------------------------------------------------

    WorkerStatistics&
    worker(
        std::size_t id);

    const WorkerStatistics&
    worker(
        std::size_t id) const;

    //----------------------------------------------------------
    // Global totals
    //----------------------------------------------------------

    std::size_t totalInstructions() const;

    std::size_t totalBlocks() const;

    std::size_t totalRegions() const;

    std::size_t totalCacheHits() const;

    std::size_t totalCacheMisses() const;

    std::size_t totalSteals() const;

    std::size_t totalIdleCycles() const;

    std::size_t totalSynchronizations() const;

    //----------------------------------------------------------
    // Worker count
    //----------------------------------------------------------

    std::size_t workerCount() const noexcept;

private:

    std::vector<WorkerStatistics> workers_;
};

} // namespace runtime
} // namespace renorm