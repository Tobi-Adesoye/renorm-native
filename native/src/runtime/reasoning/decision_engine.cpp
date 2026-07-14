#include "renorm/runtime/reasoning/decision_engine.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void DecisionEngine::decide(
    ExecutionContext& context) const
{
    //----------------------------------------------------------
    // Build execution policy
    //----------------------------------------------------------

    context.policy.action =
        choose_action(
            context);

    context.policy.priority =
        choose_priority(
            context);

    context.policy.confidence =
        compute_policy_confidence(
            context);

    context.policy.heuristic =
        context.frontier.heuristicScore;

    context.policy.prediction =
        context.frontier.predictionConfidence;

    //----------------------------------------------------------
    // Runtime result
    //----------------------------------------------------------

    context.result.executionPriority =
        static_cast<float>(
            context.policy.priority);

    context.result.confidence =
        context.policy.confidence;
}

//==============================================================

ExecutionAction
DecisionEngine::choose_action(
    const ExecutionContext& context) const
{
    //----------------------------------------------------------
    // Goal reached
    //----------------------------------------------------------

    if (context.goalReached)
        return ExecutionAction::Delay;

    //----------------------------------------------------------
    // Heavy pruning
    //----------------------------------------------------------

    if (context.pruningRatio > 0.60f)
        return ExecutionAction::Prune;

    //----------------------------------------------------------
    // High branching
    //----------------------------------------------------------

    if (context.frontier.averageBranchingFactor >
        4.0f)
    {
        return ExecutionAction::Parallelize;
    }

    //----------------------------------------------------------
    // Weak prediction
    //----------------------------------------------------------

    if (context.frontier.predictionConfidence <
        0.25f)
    {
        return ExecutionAction::Delay;
    }

    //----------------------------------------------------------
    // Default
    //----------------------------------------------------------

    return ExecutionAction::Execute;
}

//==============================================================

std::size_t
DecisionEngine::choose_priority(
    const ExecutionContext& context) const
{
    float priority = 0.0f;

    priority +=
        context.frontier.executionPriority;

    priority +=
        context.frontier.predictionConfidence;

    priority +=
        context.executionEfficiency;

    priority /= 3.0f;

    priority =
        std::clamp(
            priority,
            0.0f,
            1.0f);

    return static_cast<std::size_t>(
        priority * 100.0f);
}

//==============================================================

float
DecisionEngine::compute_policy_confidence(
    const ExecutionContext& context) const
{
    float confidence = 0.0f;

    confidence +=
        context.frontier.predictionConfidence;

    confidence +=
        context.frontier.heuristicScore;

    confidence +=
        context.executionEfficiency;

    confidence +=
        context.workloadBalance;

    confidence /= 4.0f;

    return std::clamp(
        confidence,
        0.0f,
        1.0f);
}

} // namespace runtime
} // namespace renorm