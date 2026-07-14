#include "renorm/runtime/dead_region_predictor.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void DeadRegionPredictor::predict(
    ExecutionContext& context) const
{
    if (!context.graph)
        return;

    std::size_t deadRegions = 0;

    //----------------------------------------------------------
    // Evaluate every region
    //----------------------------------------------------------

    for (std::size_t regionId = 0;
         regionId < context.graph->compiledRegions.size();
         ++regionId)
    {
        auto& region =
            context.graph->compiledRegions[
                regionId];

        if (is_dead(
                context,
                regionId))
        {
            region.active = false;

            ++deadRegions;
        }
    }

    //----------------------------------------------------------
    // Runtime statistics
    //----------------------------------------------------------

    if (context.graph->regionCount > 0)
    {
        context.pruningRatio =
            static_cast<float>(deadRegions) /
            static_cast<float>(
                context.graph->regionCount);
    }

    context.result.expansionReduction =
        context.pruningRatio;
}

//==============================================================

bool DeadRegionPredictor::is_dead(
    const ExecutionContext& context,
    std::size_t regionId) const
{
    return
        dead_score(
            context,
            regionId)
        < 0.25f;
}

//==============================================================

float DeadRegionPredictor::dead_score(
    const ExecutionContext& context,
    std::size_t regionId) const
{
    const auto& region =
        context.graph->compiledRegions[
            regionId];

    float score = 0.0f;

    //----------------------------------------------------------
    // Region confidence
    //----------------------------------------------------------

    score += region.confidence;

    //----------------------------------------------------------
    // Region heuristic
    //----------------------------------------------------------

    score += region.heuristic;

    //----------------------------------------------------------
    // Density
    //----------------------------------------------------------

    score += region.density;

    //----------------------------------------------------------
    // Connectivity
    //----------------------------------------------------------

    score +=
        std::min(
            1.0f,
            static_cast<float>(
                region.neighbouringRegions.size())
            * 0.10f);

    //----------------------------------------------------------
    // Frontier confidence
    //----------------------------------------------------------

    score +=
        context.frontier.predictionConfidence;

    //----------------------------------------------------------
    // Normalize
    //----------------------------------------------------------

    score /= 5.0f;

    return std::clamp(
        score,
        0.0f,
        1.0f);
}

} // namespace runtime
} // namespace renorm