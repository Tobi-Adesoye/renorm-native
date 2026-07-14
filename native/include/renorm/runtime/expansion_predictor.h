#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// ExpansionPredictor
//
// Predicts:
//
//  • next frontier size
//  • expansion cost
//  • branch expansion
//  • floating-point workload
//
// ExecutionContext
//        │
//        ▼
// ExpansionPredictor
//==============================================================

class ExpansionPredictor
{
public:

    ExpansionPredictor() = default;

    //----------------------------------------------------------
    // Execute prediction
    //----------------------------------------------------------

    void predict(
        ExecutionContext& context) const;

private:

    float estimate_frontier(
        const ExecutionContext& context) const;

    float estimate_cost(
        const ExecutionContext& context) const;

    float estimate_branching(
        const ExecutionContext& context) const;

    float estimate_flops(
        const ExecutionContext& context) const;
};

}
}