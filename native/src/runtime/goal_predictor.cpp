#include "renorm/runtime/goal_predictor.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void GoalPredictor::predict(
    ExecutionContext& context) const
{
    //----------------------------------------------------------
    // Predict remaining distance
    //----------------------------------------------------------

    context.frontier.predictedGoalDistance =
        estimate_distance(
            context);

    //----------------------------------------------------------
    // Predict confidence
    //----------------------------------------------------------

    context.frontier.predictionConfidence =
        estimate_confidence(
            context);

    //----------------------------------------------------------
    // Runtime result
    //----------------------------------------------------------

    context.result.predictedGoalDistance =
        context.frontier.predictedGoalDistance;

    context.result.predictionConfidence =
        context.frontier.predictionConfidence;

    context.result.predictionAccuracy =
        estimate_probability(
            context);
}

//==============================================================

float GoalPredictor::estimate_distance(
    const ExecutionContext& context) const
{
    if (context.frontier.goalReached)
        return 0.0f;

    float active =
        static_cast<float>(
            context.frontier.active.size());

    float branching =
        std::max(
            1.0f,
            context.frontier.averageBranchingFactor);

    float confidence =
        std::max(
            0.10f,
            context.frontier.predictionConfidence);

    float distance =
        active /
        (branching * confidence);

    return std::max(
        0.0f,
        distance);
}

//==============================================================

float GoalPredictor::estimate_confidence(
    const ExecutionContext& context) const
{
    float confidence = 0.0f;

    confidence +=
        context.executionEfficiency;

    confidence +=
        context.workloadBalance;

    confidence +=
        context.frontier.frontierQuality;

    confidence +=
        (1.0f -
         context.pruningRatio);

    confidence /=
        4.0f;

    return std::clamp(
        confidence,
        0.0f,
        1.0f);
}

//==============================================================

float GoalPredictor::estimate_probability(
    const ExecutionContext& context) const
{
    float probability =
        context.frontier.predictionConfidence;

    probability *=
        (1.0f -
         std::min(
             1.0f,
             context.frontier.predictedGoalDistance /
             100.0f));

    return std::clamp(
        probability,
        0.0f,
        1.0f);
}

} // namespace runtime
} // namespace renorm