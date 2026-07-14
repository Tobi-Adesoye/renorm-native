#include "renorm/runtime/adaptive_scheduler.h"

#include "renorm/runtime/lock_free_queue.h"

#include <limits>

namespace renorm
{
namespace runtime
{

//==============================================================

AdaptiveScheduler::AdaptiveScheduler(
    std::size_t workerCount)
    :
    workerCount_(workerCount)
{
    queues_.reserve(workerCount_);

    for (std::size_t i = 0;
         i < workerCount_;
         ++i)
    {
        queues_.emplace_back(4096);
    }
}

//==============================================================

void AdaptiveScheduler::initialize()
{
    for (auto& queue : queues_)
    {
        std::size_t value;

        while (queue.pop(value))
        {
        }
    }
}

//==============================================================

void AdaptiveScheduler::schedule(
    const std::vector<std::size_t>& regions)
{
    for (auto region : regions)
    {
        const auto worker =
            least_loaded_worker();

        queues_[worker].push(region);
    }
}

//==============================================================

bool AdaptiveScheduler::pop(
    std::size_t workerId,
    std::size_t& region)
{
    return queues_[workerId].pop(region);
}

//==============================================================

bool AdaptiveScheduler::push(
    std::size_t workerId,
    std::size_t region)
{
    return queues_[workerId].push(region);
}

//==============================================================

std::size_t AdaptiveScheduler::queue_size(
    std::size_t workerId) const
{
    return queues_[workerId].size();
}

//==============================================================

std::size_t
AdaptiveScheduler::least_loaded_worker() const
{
    std::size_t worker = 0;

    std::size_t smallest =
        std::numeric_limits<std::size_t>::max();

    for (std::size_t i = 0;
         i < workerCount_;
         ++i)
    {
        const auto size =
            queues_[i].size();

        if (size < smallest)
        {
            smallest = size;
            worker = i;
        }
    }

    return worker;
}

} // namespace runtime
} // namespace renorm