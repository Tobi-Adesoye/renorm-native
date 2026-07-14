#include "renorm/runtime/runtime_execution_engine.h"
#include "renorm/runtime/execution_context.h"

// Subsystem Dependencies (Isolated via PIMPL)
#include "renorm/runtime/parallel_region_scheduler.h"
#include "renorm/runtime/parallel_block_executor.h"
#include "renorm/runtime/prediction_runtime.h"
#include "renorm/runtime/reasoning/reasoning_engine.h"
#include "renorm/runtime/runtime_optimizer.h"
#include "renorm/runtime/adaptive_pruner.h"
#include "renorm/runtime/runtime_result_builder.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// Implementation Struct
//==============================================================

struct RuntimeExecutionEngine::Impl
{
    // Exception-safe RAII-managed members
    ParallelRegionScheduler scheduler;
    ParallelBlockExecutor   executor;
    PredictionRuntime       prediction;
    ReasoningEngine         reasoning;
    RuntimeOptimizer        optimizer;
    AdaptivePruner          pruner;
};

//==============================================================
// Lifecycle
//==============================================================

RuntimeExecutionEngine::RuntimeExecutionEngine()
    : impl_(std::make_unique<Impl>())
{
}

RuntimeExecutionEngine::~RuntimeExecutionEngine() = default;

RuntimeExecutionEngine::RuntimeExecutionEngine(RuntimeExecutionEngine&&) noexcept = default;

RuntimeExecutionEngine& RuntimeExecutionEngine::operator=(RuntimeExecutionEngine&&) noexcept = default;

//==============================================================
// Execution Pipeline
//==============================================================

void RuntimeExecutionEngine::execute(ExecutionContext& context) const
{
    initialize(context);

    while (!context.finished())
    {
        schedule_regions(context);

        execute_blocks(context);

        execute_prediction(context);

        execute_reasoning(context);

        optimize(context);

        prune(context);

        ++context.iteration;

        if (should_terminate(context))
        {
            context.terminate = true;
        }
    }

    finalize(context);
}

//==============================================================
// Pipeline Stages & Policy Helpers
//==============================================================

void RuntimeExecutionEngine::initialize(ExecutionContext& context) const
{
    // Verify graph is retained before state initialization.
    // context.clear_runtime() safely resets runtime fields while 
    // leaving the immutable context.graph pointer unperturbed.
    context.clear_runtime();
}

void RuntimeExecutionEngine::schedule_regions(ExecutionContext& context) const
{
    impl_->scheduler.schedule(context);
}

void RuntimeExecutionEngine::execute_blocks(ExecutionContext& context) const
{
    impl_->executor.execute(context);
}

void RuntimeExecutionEngine::execute_prediction(ExecutionContext& context) const
{
    impl_->prediction.predict(context);
}

void RuntimeExecutionEngine::execute_reasoning(ExecutionContext& context) const
{
    impl_->reasoning.execute(context);
}

void RuntimeExecutionEngine::optimize(ExecutionContext& context) const
{
    impl_->optimizer.optimize(context);
}

void RuntimeExecutionEngine::prune(ExecutionContext& context) const
{
    impl_->pruner.prune(context);
}

void RuntimeExecutionEngine::finalize(ExecutionContext& context) const
{
    RuntimeResultBuilder::build(context);
}

bool RuntimeExecutionEngine::should_terminate(const ExecutionContext& context) const
{
    // Encapsulated criteria allows future updates (timeouts, etc.) without altering main loop
    return context.regionFrontier.empty();
}

} // namespace runtime
} // namespace renorm