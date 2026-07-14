#include "renorm/runtime/region_scheduler.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

void RegionScheduler::schedule(
    ExecutionContext& context)
{
    compute_levels(context);

    assign_priorities(context);

    build_execution_queue(context);
}

//==============================================================

void RegionScheduler::compute_levels(
    ExecutionContext& context)
{
    auto& schedule =
        context.regionSchedule;

    schedule.clear();

    for (std::size_t i = 0;
         i < context.regionPath.size();
         ++i)
    {
        schedule.push_back(
        {
            context.regionPath[i],
            i,
            0.0f
        });
    }
}

//==============================================================

void RegionScheduler::assign_priorities(
    ExecutionContext& context)
{
    for (auto& item :
         context.regionSchedule)
    {
        auto& region =
            context.compiled
                .regions[item.region];

        item.priority =
            region.confidence
            + region.heuristic
            + region.density;
    }
}

//==============================================================

void RegionScheduler::build_execution_queue(
    ExecutionContext& context)
{
    std::sort(
        context.regionSchedule.begin(),
        context.regionSchedule.end(),
        [](const auto& a,
           const auto& b)
        {
            return a.priority >
                   b.priority;
        });
}

}
}