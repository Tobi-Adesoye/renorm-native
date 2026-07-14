#include "renorm/runtime/adaptive_execution_policy.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void AdaptiveExecutionPolicy::update(
    ExecutionContext& context) const
{
    context.policy.confidence =
        compute_confidence(context);

    context.policy.heuristic =
        compute_heuristic(context);

    context.policy.prediction =
        compute_prediction(context);

    context.policy.priority =
        compute_priority(context);

    context.policy.action =
        choose_action(context);

    //----------------------------------------------------------
    // Mirror into RuntimeResult
    //----------------------------------------------------------

    context.result.confidence =
        context.policy.confidence;

    context.result.heuristicScore =
        context.policy.heuristic;

    context.result.predictionConfidence =
        context.policy.prediction;

    context.result.executionPriority =
        static_cast<float>(
            context.policy.priority);
}

//==============================================================

float AdaptiveExecutionPolicy::compute_confidence(
    const ExecutionContext& context) const
{
    float confidence = 1.0f;

    confidence *=
        std::max(
            0.0f,
            1.0f -
            context.pruningRatio);

    confidence *=
        context.executionEfficiency;

    confidence *=
        context.workloadBalance;

    return std::clamp(
        confidence,
        0.0f,
        1.0f);
}

//==============================================================

float AdaptiveExecutionPolicy::compute_heuristic(
    const ExecutionContext& context) const
{
    float score = 0.0f;

    score +=
        context.frontier.heuristicScore;

    score +=
        context.frontier.frontierQuality;

    score +=
        context.frontier.graphComplexity;

    return score / 3.0f;
}

//==============================================================

float AdaptiveExecutionPolicy::compute_prediction(
    const ExecutionContext& context) const
{
    return context.frontier.predictionConfidence;
}

//==============================================================

std::size_t AdaptiveExecutionPolicy::compute_priority(
    const ExecutionContext& context) const
{
    std::size_t priority = 0;

    priority +=
        context.executedRegions;

    priority +=
        context.executedBlocks;

    priority +=
        context.frontier.activeBlocks.size();

    return priority;
}

//==============================================================

ExecutionAction AdaptiveExecutionPolicy::choose_action(
    const ExecutionContext& context) const
{
    //----------------------------------------------------------
    // Heavy pruning
    //----------------------------------------------------------

    if (context.pruningRatio > 0.60f)
        return ExecutionAction::Prune;

    //----------------------------------------------------------
    // Multiple workers available
    //----------------------------------------------------------

    if (context.workerCount > 1 &&
        context.frontier.activeBlocks.size() >
        context.workerCount)
    {
        return ExecutionAction::Parallelize;
    }

    //----------------------------------------------------------
    // Poor confidence
    //----------------------------------------------------------

    if (context.policy.confidence < 0.30f)
        return ExecutionAction::Delay;

    //----------------------------------------------------------
    // Normal execution
    //----------------------------------------------------------

    return ExecutionAction::Execute;
}

} // namespace runtime
} // namespace renorm