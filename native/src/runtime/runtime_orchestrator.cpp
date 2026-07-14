#include "renorm/runtime/runtime_orchestrator.h"

#include "renorm/runtime/execution_context.h"

#include "renorm/runtime/runtime_optimizer.h"
#include "renorm/runtime/runtime_specializer.h"
#include "renorm/runtime/execution_plan_optimizer.h"

#include "renorm/runtime/cache_locality_optimizer.h"
#include "renorm/runtime/memory_prefetcher.h"

#include "renorm/runtime/adaptive_scheduler.h"
#include "renorm/runtime/load_balancer.h"

#include "renorm/runtime/runtime_execution_engine.h"

#include "renorm/runtime/reasoning/reasoning_engine.h"

#include "renorm/runtime/adaptive_pruner.h"

#include "renorm/runtime/runtime_profiler.h"

namespace renorm
{
namespace runtime
{

//==============================================================

struct RuntimeOrchestrator::Impl
{
    RuntimeOptimizer optimizer;

    RuntimeSpecializer specializer;

    ExecutionPlanOptimizer executionPlan;

    CacheLocalityOptimizer cacheOptimizer;

    MemoryPrefetcher prefetcher;

    AdaptiveScheduler scheduler;

    LoadBalancer loadBalancer;

    RuntimeExecutionEngine execution;

    ReasoningEngine reasoning;

    AdaptivePruner pruner;

    RuntimeProfiler profiler;
};

//==============================================================

RuntimeOrchestrator::RuntimeOrchestrator()
    : impl_(std::make_unique<Impl>())
{
}

RuntimeOrchestrator::~RuntimeOrchestrator() = default;

RuntimeOrchestrator::RuntimeOrchestrator(
    RuntimeOrchestrator&&) noexcept = default;

RuntimeOrchestrator&
RuntimeOrchestrator::operator=(
    RuntimeOrchestrator&&) noexcept = default;

//==============================================================

void RuntimeOrchestrator::execute(
    ExecutionContext& context)
{
    initialize(context);

    optimize(context);

    specialize(context);

    build_execution_plan(context);

    prepare_runtime(context);

    execute_runtime(context);

    reasoning(context);

    prune(context);

    finalize(context);
}

//==============================================================

void RuntimeOrchestrator::initialize(
    ExecutionContext& context)
{
    impl_->profiler.reset();

    context.clear_runtime();
}

//==============================================================

void RuntimeOrchestrator::optimize(
    ExecutionContext& context)
{
    impl_->profiler.begin(
        RuntimeProfiler::Stage::RuntimeOptimization);

    impl_->optimizer.optimize(
        context);

    impl_->profiler.end(
        RuntimeProfiler::Stage::RuntimeOptimization);
}

//==============================================================

void RuntimeOrchestrator::specialize(
    ExecutionContext& context)
{
    impl_->profiler.begin(
        RuntimeProfiler::Stage::RuntimeSpecialization);

    impl_->specializer.specialize(
        context);

    impl_->profiler.end(
        RuntimeProfiler::Stage::RuntimeSpecialization);
}

//==============================================================

void RuntimeOrchestrator::build_execution_plan(
    ExecutionContext& context)
{
    impl_->executionPlan.optimize(
        context);
}

//==============================================================

void RuntimeOrchestrator::prepare_runtime(
    ExecutionContext& context)
{
    impl_->cacheOptimizer.optimize(
        context);

    impl_->prefetcher.prefetch(
        context);

    impl_->scheduler.schedule(
        context);

    impl_->loadBalancer.balance(
        context);
}

//==============================================================

void RuntimeOrchestrator::execute_runtime(
    ExecutionContext& context)
{
    impl_->profiler.begin(
        RuntimeProfiler::Stage::ParallelExecution);

    impl_->execution.execute(
        context);

    impl_->profiler.end(
        RuntimeProfiler::Stage::ParallelExecution);
}

//==============================================================

void RuntimeOrchestrator::reasoning(
    ExecutionContext& context)
{
    impl_->profiler.begin(
        RuntimeProfiler::Stage::Reasoning);

    impl_->reasoning.execute(
        context);

    impl_->profiler.end(
        RuntimeProfiler::Stage::Reasoning);
}

//==============================================================

void RuntimeOrchestrator::prune(
    ExecutionContext& context)
{
    impl_->profiler.begin(
        RuntimeProfiler::Stage::AdaptivePruning);

    impl_->pruner.prune(
        context);

    impl_->profiler.end(
        RuntimeProfiler::Stage::AdaptivePruning);
}

//==============================================================

void RuntimeOrchestrator::finalize(
    ExecutionContext& context)
{
    impl_->profiler.begin(
        RuntimeProfiler::Stage::Finalization);

    impl_->profiler.update(
        context);

    context.result.executionTimeMs =
        impl_->profiler.total_runtime_ms();

    impl_->profiler.end(
        RuntimeProfiler::Stage::Finalization);
}

} // namespace runtime
} // namespace renorm