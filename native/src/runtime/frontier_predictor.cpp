#include "renorm/runtime/frontier_predictor.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

void FrontierPredictor::predict(
    ExecutionContext& context)
{
    float total = 0.0f;

    for (auto& region : context.compiled.regions)
    {
        region.predictionScore =
            compute_score(region);

        total +=
            region.predictionScore;
    }

    if (total <= 0.0f)
        return;

    normalize(context);
}

//==============================================================

float FrontierPredictor::compute_score(
    const CompiledRegion& region) const
{
    float score = 0.0f;

    score += region.heuristic;

    score += region.confidence;

    score += region.density;

    score +=
        static_cast<float>(
            region.neighbouringRegions.size());

    return score;
}

//==============================================================

void FrontierPredictor::normalize(
    ExecutionContext& context)
{
    float total = 0.0f;

    for (const auto& region :
         context.compiled.regions)
    {
        total += region.predictionScore;
    }

    if (total <= 0.0f)
        return;

    for (auto& region :
         context.compiled.regions)
    {
        region.predictionProbability =
            region.predictionScore /
            total;
    }
}

}
}