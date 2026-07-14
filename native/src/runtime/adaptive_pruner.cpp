#include "renorm/runtime/adaptive_pruner.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void AdaptivePruner::update(
    ExecutionContext& context) const
{
    if (context.schedule.empty())
        return;

    const std::size_t originalSize =
        context.schedule.size();

    auto end =
        std::remove_if(
            context.schedule.begin(),
            context.schedule.end(),
            [&](const RegionScheduleItem& item)
            {
                return should_prune(
                    context,
                    item.region);
            });

    context.schedule.erase(
        end,
        context.schedule.end());

    //----------------------------------------------------------
    // Runtime statistics
    //----------------------------------------------------------

    if (originalSize > 0)
    {
        context.pruningRatio =
            1.0f -
            (
                static_cast<float>(
                    context.schedule.size())
                /
                static_cast<float>(
                    originalSize)
            );
    }
    else
    {
        context.pruningRatio = 0.0f;
    }

    context.result.expansionReduction =
        context.pruningRatio;

    context.result.executionPriority =
        static_cast<float>(
            context.schedule.size());
}

//==============================================================

bool AdaptivePruner::should_prune(
    const ExecutionContext& context,
    std::size_t regionId) const
{
    return
        score(
            context,
            regionId)
        <
        threshold(
            context);
}

//==============================================================

float AdaptivePruner::score(
    const ExecutionContext& context,
    std::size_t regionId) const
{
    const auto& region =
        context.graph->compiledRegions[
            regionId];

    float value = 0.0f;

    //----------------------------------------------------------
    // Region metrics
    //----------------------------------------------------------

    value += region.density;
    value += region.heuristic;
    value += region.confidence;

    //----------------------------------------------------------
    // Runtime metrics
    //----------------------------------------------------------

    value +=
        context.frontier.frontierDensity;

    value +=
        context.frontier.predictionConfidence;

    value +=
        context.executionEfficiency;

    return value / 6.0f;
}

//==============================================================

float AdaptivePruner::threshold(
    const ExecutionContext& context) const
{
    float t = 0.20f;

    //----------------------------------------------------------
    // Increase pruning when frontier explodes
    //----------------------------------------------------------

    if (context.frontier.frontierDensity > 0.50f)
        t += 0.15f;

    //----------------------------------------------------------
    // Poor prediction confidence
    //----------------------------------------------------------

    if (context.frontier.predictionConfidence < 0.40f)
        t += 0.10f;

    //----------------------------------------------------------
    // Low execution efficiency
    //----------------------------------------------------------

    if (context.executionEfficiency < 0.50f)
        t += 0.10f;

    return std::clamp(
        t,
        0.0f,
        1.0f);
}

} // namespace runtime
} // namespace renorm