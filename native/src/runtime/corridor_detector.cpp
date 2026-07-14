#include "renorm/runtime/corridor_detector.h"

namespace renorm
{
namespace runtime
{

void CorridorDetector::detect(
    ExecutionContext& context)
{
    context.statistics.corridorRegions = 0;
    context.statistics.deadEndRegions = 0;
    context.statistics.linearRegions = 0;

    for (auto& region : context.compiled.regions)
    {
        if (is_corridor(region, context))
        {
            ++context.statistics.corridorRegions;
        }

        if (is_dead_end(region))
        {
            ++context.statistics.deadEndRegions;
        }

        if (is_linear(region))
        {
            ++context.statistics.linearRegions;
        }
    }
}

//==============================================================

bool CorridorDetector::is_corridor(
    const CompiledRegion& region,
    const ExecutionContext&) const
{
    return region.neighbouringRegions.size() == 2;
}

//==============================================================

bool CorridorDetector::is_dead_end(
    const CompiledRegion& region) const
{
    return region.neighbouringRegions.size() <= 1;
}

//==============================================================

bool CorridorDetector::is_linear(
    const CompiledRegion& region) const
{
    return
        region.blocks.size() == 1 &&
        region.neighbouringRegions.size() == 2;
}

}
}