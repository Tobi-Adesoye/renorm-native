#include "renorm/runtime/hub_detector.h"

namespace renorm
{
namespace runtime
{

void HubDetector::detect(
    ExecutionContext& context)
{
    context.statistics.hubRegions = 0;

    const float average =
        compute_average_degree(context);

    for (auto& region : context.compiled.regions)
    {
        const float degree =
            static_cast<float>(
                region.neighbouringRegions.size());

        if (degree >= average * 2.0f)
        {
            region.heuristic += 2.0f;

            region.confidence += 0.20f;

            ++context.statistics.hubRegions;
        }
    }
}

//==============================================================

float HubDetector::compute_average_degree(
    const ExecutionContext& context) const
{
    if (context.compiled.regions.empty())
        return 0.0f;

    std::size_t total = 0;

    for (const auto& region :
         context.compiled.regions)
    {
        total +=
            region.neighbouringRegions.size();
    }

    return
        static_cast<float>(total) /
        static_cast<float>(
            context.compiled.regions.size());
}

}
}