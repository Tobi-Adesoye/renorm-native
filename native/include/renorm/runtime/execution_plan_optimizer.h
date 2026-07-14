#pragma once

#include <cstddef>

namespace renorm
{
namespace runtime
{

class ExecutionContext;

//==============================================================
//
// ExecutionPlanOptimizer
//
// Produces a specialized execution plan from runtime state.
//
// Responsibilities
//
// • Block fusion
// • Region fusion
// • SIMD specialization
// • Branch elimination
// • Execution ordering
// • Memory ordering
//
//==============================================================

class ExecutionPlanOptimizer
{
public:

    ExecutionPlanOptimizer() = default;

    //----------------------------------------------------------
    // Optimize execution plan
    //----------------------------------------------------------

    void optimize(
        ExecutionContext& context) const;

private:

    void build_execution_plan(
        ExecutionContext& context) const;

    void optimize_block_order(
        ExecutionContext& context) const;

    void optimize_region_order(
        ExecutionContext& context) const;

    void optimize_memory_layout(
        ExecutionContext& context) const;

    void optimize_vectorization(
        ExecutionContext& context) const;

    void optimize_branches(
        ExecutionContext& context) const;

    void optimize_prefetch(
        ExecutionContext& context) const;
};

}
}