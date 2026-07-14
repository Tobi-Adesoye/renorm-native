#include "renorm/runtime/reasoning/prediction_engine.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

namespace
{
    //----------------------------------------------------------
    // Prediction tuning constants
    //----------------------------------------------------------

    constexpr float kGoalBias = 0.25f;

    constexpr float kMinBranching = 1.0f;

    constexpr float kConfidenceNormalization = 4.0f;
}

//==============================================================

void PredictionEngine::predict(
    ExecutionContext& context) const
{
    predict_frontier(context);

    predict_expansion(context);

    predict_goal(context);

    predict_confidence(context);
}

//==============================================================

void PredictionEngine::predict_frontier(
    ExecutionContext& context) const
{
    auto& frontier = context.frontier;
    auto& result   = context.result;

    //----------------------------------------------------------
    // Future SIMD runtime will migrate to regionFrontier.
    // Until then we continue using active node count.
    //----------------------------------------------------------

    float predicted =
        static_cast<float>(
            frontier.active.size());

    predicted *=
        std::max(
            kMinBranching,
            frontier.averageBranchingFactor);

    predicted *=
        (1.0f - context.pruningRatio);

    predicted *=
        frontier.frontierQuality;

    frontier.predictedFrontierSize = predicted;
    result.predictedFrontierSize   = predicted;
}

//==============================================================

void PredictionEngine::predict_expansion(
    ExecutionContext& context) const
{
    auto& frontier = context.frontier;
    auto& result   = context.result;

    float expansion =
        frontier.predictedFrontierSize +
        frontier.frontierDensity +
        frontier.graphComplexity;

    expansion *=
        context.executionEfficiency;

    frontier.predictedExpansionCost =
        expansion;

    frontier.predictedBranchExpansion =
        expansion *
        std::max(
            kMinBranching,
            frontier.averageBranchingFactor);

    result.predictedExpansionCost =
        frontier.predictedExpansionCost;

    result.predictedBranchExpansion =
        frontier.predictedBranchExpansion;
}

//==============================================================

void PredictionEngine::predict_goal(
    ExecutionContext& context) const
{
    auto& frontier = context.frontier;
    auto& result   = context.result;

    if (frontier.goalReached)
    {
        frontier.predictedGoalDistance = 0.0f;
    }
    else
    {
        float distance =
            frontier.predictedFrontierSize;

        distance /=
            std::max(
                kMinBranching,
                frontier.averageBranchingFactor);

        distance *=
            (1.0f -
             frontier.predictionConfidence +
             kGoalBias);

        frontier.predictedGoalDistance =
            distance;
    }

    result.predictedGoalDistance =
        frontier.predictedGoalDistance;
}

//==============================================================

void PredictionEngine::predict_confidence(
    ExecutionContext& context) const
{
    auto& frontier = context.frontier;
    auto& result   = context.result;

    float confidence = 0.0f;

    confidence += frontier.frontierQuality;
    confidence += context.executionEfficiency;
    confidence += context.workloadBalance;
    confidence += (1.0f - context.pruningRatio);

    confidence /=
        kConfidenceNormalization;

    confidence =
        std::clamp(
            confidence,
            0.0f,
            1.0f);

    frontier.predictionConfidence =
        confidence;

    result.predictionConfidence =
        confidence;

    result.predictionAccuracy =
        confidence;
}

} // namespace runtime
} // namespace renorm