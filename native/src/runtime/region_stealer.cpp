#include "renorm/runtime/region_stealer.h"

namespace renorm
{
namespace runtime
{

//==============================================================

RegionStealer::RegionStealer(
    std::size_t workerCount)
{
    if (workerCount == 0)
        workerCount = 1;

    workers_.resize(workerCount);
}

//==============================================================

void RegionStealer::initialize(
    const std::vector<std::size_t>& regions)
{
    //----------------------------------------------------------
    // Clear queues
    //----------------------------------------------------------

    for (auto& worker : workers_)
    {
        std::lock_guard<std::mutex> lock(
            worker.mutex);

        worker.queue.clear();
    }

    //----------------------------------------------------------
    // Round-robin distribution
    //----------------------------------------------------------

    std::size_t worker = 0;

    for (std::size_t region : regions)
    {
        workers_[worker].queue.push_back(
            region);

        worker =
            (worker + 1) %
            workers_.size();
    }

    remaining_.store(
        regions.size(),
        std::memory_order_release);
}

//==============================================================

std::optional<std::size_t>
RegionStealer::pop_local(
    std::size_t workerId)
{
    WorkerQueue& worker =
        workers_[workerId];

    std::lock_guard<std::mutex> lock(
        worker.mutex);

    if (worker.queue.empty())
        return std::nullopt;

    std::size_t region =
        worker.queue.front();

    worker.queue.pop_front();

    remaining_.fetch_sub(
        1,
        std::memory_order_acq_rel);

    return region;
}

//==============================================================

std::optional<std::size_t>
RegionStealer::steal(
    std::size_t workerId)
{
    const std::size_t count =
        workers_.size();

    for (std::size_t i = 0;
         i < count;
         ++i)
    {
        const std::size_t victim =
            (workerId + i + 1) % count;

        WorkerQueue& queue =
            workers_[victim];

        std::lock_guard<std::mutex> lock(
            queue.mutex);

        if (queue.queue.empty())
            continue;

        std::size_t region =
            queue.queue.back();

        queue.queue.pop_back();

        remaining_.fetch_sub(
            1,
            std::memory_order_acq_rel);

        return region;
    }

    return std::nullopt;
}

//==============================================================

bool RegionStealer::empty() const
{
    return
        remaining_.load(
            std::memory_order_acquire) == 0;
}

} // namespace runtime
} // namespace renorm