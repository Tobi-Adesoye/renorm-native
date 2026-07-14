#include "renorm/runtime/execution_plan_optimizer.h"

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void ExecutionPlanOptimizer::optimize(
    ExecutionContext& context) const
{
    build_execution_plan(context);

    optimize_block_order(context);

    optimize_region_order(context);

    optimize_memory_layout(context);

    optimize_vectorization(context);

    optimize_branches(context);

    optimize_prefetch(context);
}

//==============================================================

void ExecutionPlanOptimizer::build_execution_plan(
    ExecutionContext& context) const
{
    context.executionPlanValid = true;
}

//==============================================================

void ExecutionPlanOptimizer::optimize_block_order(
    ExecutionContext& context) const
{
    context.blockOrderOptimized = true;
}

//==============================================================

void ExecutionPlanOptimizer::optimize_region_order(
    ExecutionContext& context) const
{
    context.regionOrderOptimized = true;
}

//==============================================================

void ExecutionPlanOptimizer::optimize_memory_layout(
    ExecutionContext& context) const
{
    context.memoryLayoutOptimized = true;
}

//==============================================================

void ExecutionPlanOptimizer::optimize_vectorization(
    ExecutionContext& context) const
{
    if (context.simdAvailable)
    {
        context.vectorPlanEnabled = true;
    }
}

//==============================================================

void ExecutionPlanOptimizer::optimize_branches(
    ExecutionContext& context) const
{
    if (context.frontier.predictionConfidence > 0.80f)
    {
        context.branchPredictionEnabled = true;
    }
}

//==============================================================

void ExecutionPlanOptimizer::optimize_prefetch(
    ExecutionContext& context) const
{
    context.prefetchEnabled = true;
}

} // namespace runtime
} // namespace renorm