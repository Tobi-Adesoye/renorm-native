#include "renorm/runtime/reasoning/heuristic_engine.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void HeuristicEngine::compute(
    ExecutionContext& context) const
{
    //----------------------------------------------------------
    // Individual heuristics
    //----------------------------------------------------------

    const float graph =
        compute_graph_heuristic(
            context);

    const float frontier =
        compute_frontier_heuristic(
            context);

    const float prediction =
        compute_prediction_heuristic(
            context);

    //----------------------------------------------------------
    // Overall heuristic
    //----------------------------------------------------------

    float heuristic =
        (graph +
         frontier +
         prediction) / 3.0f;

    heuristic =
        std::clamp(
            heuristic,
            0.0f,
            1.0f);

    context.frontier.heuristicScore =
        heuristic;

    context.result.heuristicScore =
        heuristic;

    //----------------------------------------------------------
    // Execution priority
    //----------------------------------------------------------

    float priority =
        compute_execution_priority(
            context);

    context.frontier.executionPriority =
        priority;

    context.result.executionPriority =
        priority;

    //----------------------------------------------------------
    // Decision cost
    //----------------------------------------------------------

    float cost =
        1.0f - heuristic;

    cost =
        std::clamp(
            cost,
            0.0f,
            1.0f);

    context.frontier.decisionCost =
        cost;

    context.result.decisionCost =
        cost;
}

//==============================================================

float HeuristicEngine::compute_graph_heuristic(
    const ExecutionContext& context) const
{
    float value = 0.0f;

    value +=
        context.frontier.frontierDensity;

    value +=
        context.frontier.frontierQuality;

    value +=
        (1.0f /
         (1.0f +
          context.frontier.graphComplexity));

    return
        std::clamp(
            value / 3.0f,
            0.0f,
            1.0f);
}

//==============================================================

float HeuristicEngine::compute_frontier_heuristic(
    const ExecutionContext& context) const
{
    float active =
        static_cast<float>(
            context.frontier.active.size());

    float score =
        active /
        (active + 10.0f);

    score *=
        context.executionEfficiency;

    return
        std::clamp(
            score,
            0.0f,
            1.0f);
}

//==============================================================

float HeuristicEngine::compute_prediction_heuristic(
    const ExecutionContext& context) const
{
    float prediction = 0.0f;

    prediction +=
        context.frontier.predictionConfidence;

    prediction +=
        (1.0f /
         (1.0f +
          context.frontier.predictedGoalDistance));

    prediction +=
        (1.0f -
         context.pruningRatio);

    return
        std::clamp(
            prediction / 3.0f,
            0.0f,
            1.0f);
}

//==============================================================

float HeuristicEngine::compute_execution_priority(
    const ExecutionContext& context) const
{
    float priority = 0.0f;

    priority +=
        context.frontier.heuristicScore;

    priority +=
        context.executionEfficiency;

    priority +=
        context.workloadBalance;

    priority /= 3.0f;

    return
        std::clamp(
            priority,
            0.0f,
            1.0f);
}

} // namespace runtime
} // namespace renorm