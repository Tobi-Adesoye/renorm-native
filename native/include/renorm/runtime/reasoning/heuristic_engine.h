#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// HeuristicEngine
//
// Computes adaptive runtime heuristics.
//
// Consumes:
//
//   PatternDetector
//   PredictionEngine
//
// Produces:
//
//   heuristicScore
//   executionPriority
//   decisionCost
//   frontierQuality
//
//==============================================================

class HeuristicEngine
{
public:

    HeuristicEngine() = default;

    //----------------------------------------------------------

    void compute(
        ExecutionContext& context) const;

private:

    float compute_graph_heuristic(
        const ExecutionContext& context) const;

    float compute_frontier_heuristic(
        const ExecutionContext& context) const;

    float compute_prediction_heuristic(
        const ExecutionContext& context) const;

    float compute_execution_priority(
        const ExecutionContext& context) const;
};

}
}