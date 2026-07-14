#include "renorm/runtime/load_balancer.h"

#include "renorm/runtime/adaptive_scheduler.h"

#include <limits>

namespace renorm
{
namespace runtime
{

//==============================================================

LoadBalancer::LoadBalancer(
    AdaptiveScheduler& scheduler)
    :
    scheduler_(scheduler)
{
}

//==============================================================

void LoadBalancer::balance()
{
    //----------------------------------------------------------
    // Determine queue imbalance
    //----------------------------------------------------------

    const std::size_t source =
        find_most_loaded();

    const std::size_t target =
        find_least_loaded();

    if (source == target)
        return;

    //----------------------------------------------------------
    // Threshold prevents unnecessary migrations
    //----------------------------------------------------------

    const std::size_t sourceSize =
        scheduler_.queue_size(source);

    const std::size_t targetSize =
        scheduler_.queue_size(target);

    if (sourceSize <= targetSize + 1)
        return;

    //----------------------------------------------------------
    // Move one region
    //----------------------------------------------------------

    std::size_t region;

    if (scheduler_.pop(source, region))
    {
        scheduler_.push(
            target,
            region);
    }
}

//==============================================================

std::size_t
LoadBalancer::find_most_loaded() const
{
    std::size_t worker = 0;

    std::size_t largest = 0;

    for (std::size_t i = 0;; ++i)
    {
        const std::size_t size =
            scheduler_.queue_size(i);

        if (size > largest)
        {
            largest = size;
            worker = i;
        }

        //------------------------------------------------------
        // Stop when queue_size becomes invalid.
        // Scheduler should later expose worker_count().
        //------------------------------------------------------

        if (i + 1 >= scheduler_.worker_count())
            break;
    }

    return worker;
}

//==============================================================

std::size_t
LoadBalancer::find_least_loaded() const
{
    std::size_t worker = 0;

    std::size_t smallest =
        std::numeric_limits<std::size_t>::max();

    for (std::size_t i = 0;; ++i)
    {
        const std::size_t size =
            scheduler_.queue_size(i);

        if (size < smallest)
        {
            smallest = size;
            worker = i;
        }

        if (i + 1 >= scheduler_.worker_count())
            break;
    }

    return worker;
}

} // namespace runtime
} // namespace renorm