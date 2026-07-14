#include "renorm/runtime/prediction_runtime.h"

#include "renorm/runtime/region_predictor.h"
#include "renorm/runtime/expansion_predictor.h"
#include "renorm/runtime/dead_region_predictor.h"
#include "renorm/runtime/corridor_predictor.h"
#include "renorm/runtime/goal_predictor.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void PredictionRuntime::execute(
    ExecutionContext& context) const
{
    //----------------------------------------------------------
    // Predict active regions
    //----------------------------------------------------------

    RegionPredictor regionPredictor;

    regionPredictor.predict(
        context);

    //----------------------------------------------------------
    // Predict future expansion
    //----------------------------------------------------------

    ExpansionPredictor expansionPredictor;

    expansionPredictor.predict(
        context);

    //----------------------------------------------------------
    // Detect dead regions
    //----------------------------------------------------------

    DeadRegionPredictor deadPredictor;

    deadPredictor.predict(
        context);

    //----------------------------------------------------------
    // Corridor prediction
    //----------------------------------------------------------

    CorridorPredictor corridorPredictor;

    corridorPredictor.predict(
        context);

    //----------------------------------------------------------
    // Goal prediction
    //----------------------------------------------------------

    GoalPredictor goalPredictor;

    goalPredictor.predict(
        context);

    //----------------------------------------------------------
    // Final normalization
    //----------------------------------------------------------

    normalize_predictions(
        context);
}

//==============================================================

void PredictionRuntime::normalize_predictions(
    ExecutionContext& context) const
{
    //----------------------------------------------------------
    // Clamp confidence
    //----------------------------------------------------------

    context.frontier.predictionConfidence =
        std::clamp(
            context.frontier.predictionConfidence,
            0.0f,
            1.0f);

    //----------------------------------------------------------
    // Clamp frontier size
    //----------------------------------------------------------

    context.frontier.predictedFrontierSize =
        std::max(
            0.0f,
            context.frontier.predictedFrontierSize);

    //----------------------------------------------------------
    // Clamp goal distance
    //----------------------------------------------------------

    context.frontier.predictedGoalDistance =
        std::max(
            0.0f,
            context.frontier.predictedGoalDistance);

    //----------------------------------------------------------
    // Clamp expansion cost
    //----------------------------------------------------------

    context.frontier.predictedExpansionCost =
        std::max(
            0.0f,
            context.frontier.predictedExpansionCost);

    //----------------------------------------------------------
    // Clamp branch expansion
    //----------------------------------------------------------

    context.frontier.predictedBranchExpansion =
        std::max(
            0.0f,
            context.frontier.predictedBranchExpansion);

    //----------------------------------------------------------
    // Copy into RuntimeResult
    //----------------------------------------------------------

    context.result.predictedFrontierSize =
        context.frontier.predictedFrontierSize;

    context.result.predictedGoalDistance =
        context.frontier.predictedGoalDistance;

    context.result.predictedExpansionCost =
        context.frontier.predictedExpansionCost;

    context.result.predictedBranchExpansion =
        context.frontier.predictedBranchExpansion;

    context.result.predictionConfidence =
        context.frontier.predictionConfidence;
}

} // namespace runtime
} // namespace renorm