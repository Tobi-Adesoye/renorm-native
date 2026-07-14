#include "renorm/runtime/region_predictor.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void RegionPredictor::predict(
    ExecutionContext& context) const
{
    if (!context.graph)
        return;

    std::size_t predictedRegions = 0;

    float confidence = 0.0f;

    //----------------------------------------------------------
    // Evaluate every compiled region
    //----------------------------------------------------------

    for (std::size_t regionId = 0;
         regionId < context.graph->compiledRegions.size();
         ++regionId)
    {
        if (!likely_active(
                context,
                regionId))
            continue;

        ++predictedRegions;

        confidence +=
            region_score(
                context,
                regionId);
    }

    //----------------------------------------------------------
    // Normalize confidence
    //----------------------------------------------------------

    if (predictedRegions > 0)
    {
        confidence /=
            static_cast<float>(
                predictedRegions);
    }

    confidence =
        std::clamp(
            confidence,
            0.0f,
            1.0f);

    //----------------------------------------------------------
    // Store prediction
    //----------------------------------------------------------

    context.result.predictedVisited =
        predictedRegions;

    context.frontier.predictionConfidence =
        confidence;
}

//==============================================================

float RegionPredictor::region_score(
    const ExecutionContext& context,
    std::size_t regionId) const
{
    const auto& region =
        context.graph->compiledRegions[
            regionId];

    float score = 0.0f;

    //----------------------------------------------------------
    // Density
    //----------------------------------------------------------

    score += region.density;

    //----------------------------------------------------------
    // Heuristic
    //----------------------------------------------------------

    score += region.heuristic;

    //----------------------------------------------------------
    // Confidence
    //----------------------------------------------------------

    score += region.confidence;

    //----------------------------------------------------------
    // Connectivity
    //----------------------------------------------------------

    score +=
        static_cast<float>(
            region.neighbouringRegions.size())
        * 0.05f;

    //----------------------------------------------------------
    // Activity
    //----------------------------------------------------------

    if (region.active)
        score += 0.20f;

    //----------------------------------------------------------
    // Normalize
    //----------------------------------------------------------

    score /= 4.20f;

    return std::clamp(
        score,
        0.0f,
        1.0f);
}

//==============================================================

bool RegionPredictor::likely_active(
    const ExecutionContext& context,
    std::size_t regionId) const
{
    const auto& region =
        context.graph->compiledRegions[
            regionId];

    //----------------------------------------------------------
    // Already inactive
    //----------------------------------------------------------

    if (!region.active)
        return false;

    //----------------------------------------------------------
    // Empty region
    //----------------------------------------------------------

    if (region.blocks.empty())
        return false;

    //----------------------------------------------------------
    // Very weak regions
    //----------------------------------------------------------

    if (region_score(
            context,
            regionId)
        < 0.20f)
    {
        return false;
    }

    return true;
}

} // namespace runtime
} // namespace renorm