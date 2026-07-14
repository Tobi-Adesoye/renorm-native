#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// LearningEngine
//
// Learns from completed executions.
//
// Phase D (v1)
// ----------------
// Compatibility implementation.
//
// Future:
//
//  • Region success statistics
//  • Block execution statistics
//  • Prediction accuracy
//  • Heuristic refinement
//  • Learned execution policy
//==============================================================

class LearningEngine
{
public:

    LearningEngine() = default;

    //----------------------------------------------------------
    // Learn from execution
    //----------------------------------------------------------

    void update(
        ExecutionContext& context);

private:

    //----------------------------------------------------------
    // Update region statistics
    //----------------------------------------------------------

    void update_regions(
        ExecutionContext& context);

    //----------------------------------------------------------
    // Update block statistics
    //----------------------------------------------------------

    void update_blocks(
        ExecutionContext& context);

    //----------------------------------------------------------
    // Update prediction model
    //----------------------------------------------------------

    void update_predictions(
        ExecutionContext& context);

    //----------------------------------------------------------
    // Update heuristics
    //----------------------------------------------------------

    void update_heuristics(
        ExecutionContext& context);
};

}
}