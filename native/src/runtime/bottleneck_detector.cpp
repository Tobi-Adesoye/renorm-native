#include "renorm/runtime/bottleneck_detector.h"

namespace renorm
{
namespace runtime
{

void BottleneckDetector::detect(
    ExecutionContext& context)
{
    context.statistics.bottleneckRegions = 0;

    for (auto& region : context.compiled.regions)
    {
        if (is_bottleneck(region, context))
        {
            ++context.statistics.bottleneckRegions;

            region.heuristic += 3.0f;

            region.confidence += 0.30f;
        }
    }
}

//==============================================================

bool BottleneckDetector::is_bottleneck(
    const CompiledRegion& region,
    const ExecutionContext&) const
{
    //----------------------------------------------------------
    // Phase E4 heuristic.
    //
    // True articulation detection arrives in Phase F.
    //----------------------------------------------------------

    return
        region.blocks.size() <= 2 &&
        region.neighbouringRegions.size() >= 3;
}

}
}