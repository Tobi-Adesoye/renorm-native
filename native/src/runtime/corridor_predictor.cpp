#include "renorm/runtime/corridor_predictor.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void CorridorPredictor::predict(
    ExecutionContext& context) const
{
    if (!context.graph)
        return;

    std::size_t corridors = 0;
    std::size_t hubs = 0;
    std::size_t bottlenecks = 0;

    //----------------------------------------------------------
    // Analyse every compiled region
    //----------------------------------------------------------

    for (std::size_t regionId = 0;
         regionId < context.graph->compiledRegions.size();
         ++regionId)
    {
        if (is_corridor(context, regionId))
            ++corridors;

        if (is_hub(context, regionId))
            ++hubs;

        if (is_bottleneck(context, regionId))
            ++bottlenecks;
    }

    //----------------------------------------------------------
    // Store into frontier
    //----------------------------------------------------------

    context.frontier.corridorNodes =
        corridors;

    context.frontier.hubNodes =
        hubs;

    context.frontier.bottleneckNodes =
        bottlenecks;

    context.frontier.frontierDensity =
        compute_density(context);

    context.frontier.frontierQuality =
        compute_quality(context);

    context.frontier.graphComplexity =
        compute_complexity(context);

    //----------------------------------------------------------
    // Mirror into RuntimeResult
    //----------------------------------------------------------

    context.result.corridorNodes =
        corridors;

    context.result.hubNodes =
        hubs;

    context.result.bottleneckNodes =
        bottlenecks;

    context.result.frontierDensity =
        context.frontier.frontierDensity;

    context.result.frontierQuality =
        context.frontier.frontierQuality;

    context.result.graphComplexity =
        context.frontier.graphComplexity;
}

//==============================================================

float CorridorPredictor::compute_density(
    const ExecutionContext& context) const
{
    if (context.graph->regionCount == 0)
        return 0.0f;

    float total = 0.0f;

    for (const auto& region :
         context.graph->compiledRegions)
    {
        total += region.density;
    }

    return total /
           static_cast<float>(
               context.graph->regionCount);
}

//==============================================================

float CorridorPredictor::compute_quality(
    const ExecutionContext& context) const
{
    float quality = 0.0f;

    quality +=
        context.frontier.predictionConfidence;

    quality +=
        context.executionEfficiency;

    quality +=
        context.workloadBalance;

    return std::clamp(
        quality / 3.0f,
        0.0f,
        1.0f);
}

//==============================================================

float CorridorPredictor::compute_complexity(
    const ExecutionContext& context) const
{
    if (context.graph->regionCount == 0)
        return 0.0f;

    float complexity = 0.0f;

    for (const auto& region :
         context.graph->compiledRegions)
    {
        complexity +=
            static_cast<float>(
                region.neighbouringRegions.size());
    }

    complexity /=
        static_cast<float>(
            context.graph->regionCount);

    return complexity;
}

//==============================================================

bool CorridorPredictor::is_corridor(
    const ExecutionContext& context,
    std::size_t regionId) const
{
    const auto& region =
        context.graph->compiledRegions[
            regionId];

    return
        region.neighbouringRegions.size() == 2 &&
        region.density > 0.50f;
}

//==============================================================

bool CorridorPredictor::is_hub(
    const ExecutionContext& context,
    std::size_t regionId) const
{
    const auto& region =
        context.graph->compiledRegions[
            regionId];

    return
        region.neighbouringRegions.size() >= 4;
}

//==============================================================

bool CorridorPredictor::is_bottleneck(
    const ExecutionContext& context,
    std::size_t regionId) const
{
    const auto& region =
        context.graph->compiledRegions[
            regionId];

    return
        region.neighbouringRegions.size() == 1 &&
        region.confidence > 0.60f;
}

} // namespace runtime
} // namespace renorm