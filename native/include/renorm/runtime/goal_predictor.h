#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// GoalPredictor
//
// Estimates:
//
//  • remaining goal distance
//  • probability goal is nearby
//  • prediction confidence
//  • expected iterations
//
// Updates:
//
//   frontier.predictedGoalDistance
//   frontier.predictionConfidence
//==============================================================

class GoalPredictor
{
public:

    GoalPredictor() = default;

    //----------------------------------------------------------
    // Execute prediction
    //----------------------------------------------------------

    void predict(
        ExecutionContext& context) const;

private:

    float estimate_distance(
        const ExecutionContext& context) const;

    float estimate_confidence(
        const ExecutionContext& context) const;

    float estimate_probability(
        const ExecutionContext& context) const;
};

}
}