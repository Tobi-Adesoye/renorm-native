#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// AdaptiveExecutionPolicy
//
// Computes the runtime execution policy from the current
// execution state.
//
// This is evaluated every iteration.
//
// ExecutionContext
//        │
//        ▼
// AdaptiveExecutionPolicy
//        │
//        ▼
// ExecutionPolicy
//==============================================================

class AdaptiveExecutionPolicy
{
public:

    AdaptiveExecutionPolicy() = default;

    //----------------------------------------------------------
    // Update runtime execution policy
    //----------------------------------------------------------

    void update(
        ExecutionContext& context) const;

private:

    //----------------------------------------------------------
    // Internal scoring
    //----------------------------------------------------------

    float compute_confidence(
        const ExecutionContext& context) const;

    float compute_heuristic(
        const ExecutionContext& context) const;

    float compute_prediction(
        const ExecutionContext& context) const;

    std::size_t compute_priority(
        const ExecutionContext& context) const;

    ExecutionAction choose_action(
        const ExecutionContext& context) const;
};

}
}