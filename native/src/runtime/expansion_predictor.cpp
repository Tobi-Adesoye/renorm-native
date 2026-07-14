#include "renorm/runtime/expansion_predictor.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void ExpansionPredictor::predict(
    ExecutionContext& context) const
{
    //----------------------------------------------------------
    // Predict frontier growth
    //----------------------------------------------------------

    context.frontier.predictedFrontierSize =
        estimate_frontier(context);

    //----------------------------------------------------------
    // Predict expansion cost
    //----------------------------------------------------------

    context.frontier.predictedExpansionCost =
        estimate_cost(context);

    //----------------------------------------------------------
    // Predict branch expansion
    //----------------------------------------------------------

    context.frontier.predictedBranchExpansion =
        estimate_branching(context);

    //----------------------------------------------------------
    // Predicted FLOPs
    //----------------------------------------------------------

    context.result.floatingPointOperations =
        static_cast<std::size_t>(
            estimate_flops(context));

    //----------------------------------------------------------
    // Copy to RuntimeResult
    //----------------------------------------------------------

    context.result.predictedFrontierSize =
        context.frontier.predictedFrontierSize;

    context.result.predictedExpansionCost =
        context.frontier.predictedExpansionCost;

    context.result.predictedBranchExpansion =
        context.frontier.predictedBranchExpansion;
}

//==============================================================

float ExpansionPredictor::estimate_frontier(
    const ExecutionContext& context) const
{
    float frontier =
        static_cast<float>(
            context.frontier.active.size());

    frontier *=
        std::max(
            1.0f,
            context.frontier.averageBranchingFactor);

    frontier *=
        (1.0f -
         context.pruningRatio);

    return frontier;
}

//==============================================================

float ExpansionPredictor::estimate_cost(
    const ExecutionContext& context) const
{
    float cost = 0.0f;

    cost +=
        static_cast<float>(
            context.frontier.activeBlocks.size());

    cost +=
        context.frontier.averageBranchingFactor;

    cost +=
        context.frontier.frontierDensity;

    cost *=
        (1.0f -
         context.executionEfficiency + 1.0f);

    return std::max(
        0.0f,
        cost);
}

//==============================================================

float ExpansionPredictor::estimate_branching(
    const ExecutionContext& context) const
{
    float branch =
        context.frontier.averageBranchingFactor;

    branch *=
        context.frontier.frontierDensity + 1.0f;

    return std::max(
        0.0f,
        branch);
}

//==============================================================

float ExpansionPredictor::estimate_flops(
    const ExecutionContext& context) const
{
    float flops = 0.0f;

    flops +=
        static_cast<float>(
            context.frontier.activeBlocks.size());

    flops *=
        std::max(
            1.0f,
            context.frontier.averageBranchingFactor);

    flops *= 64.0f;

    return flops;
}

} // namespace runtime
} // namespace renorm