#include "renorm/runtime/runtime_profiler.h"

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================

RuntimeProfiler::RuntimeProfiler()
{
}

//==============================================================

void RuntimeProfiler::begin(
    Stage stage)
{
    timers_[stage].start =
        Clock::now();
}

//==============================================================

void RuntimeProfiler::end(
    Stage stage)
{
    auto now =
        Clock::now();

    auto elapsed =
        now -
        timers_[stage].start;

    timers_[stage].elapsedMs +=
        std::chrono::duration<double,
            std::milli>(elapsed).count();
}

//==============================================================

void RuntimeProfiler::update(
    const ExecutionContext& context)
{
    simdUtilization_ =
        context.executionEfficiency;

    cacheEfficiency_ =
        context.cacheEfficiency;

    parallelEfficiency_ =
        context.parallelEfficiency;

    reasoningCost_ =
        context.frontier.decisionCost;

    predictionCost_ =
        context.frontier.predictedExpansionCost;

    schedulingCost_ =
        context.frontier.executionPriority;
}

//==============================================================

void RuntimeProfiler::reset()
{
    timers_.clear();

    simdUtilization_ = 0.0;

    cacheEfficiency_ = 0.0;

    parallelEfficiency_ = 0.0;

    reasoningCost_ = 0.0;

    predictionCost_ = 0.0;

    schedulingCost_ = 0.0;
}

//==============================================================

double RuntimeProfiler::total_runtime_ms() const
{
    double total = 0.0;

    for (const auto& timer : timers_)
    {
        total += timer.second.elapsedMs;
    }

    return total;
}

//==============================================================

double RuntimeProfiler::stage_runtime_ms(
    Stage stage) const
{
    auto it =
        timers_.find(stage);

    if (it == timers_.end())
        return 0.0;

    return it->second.elapsedMs;
}

//==============================================================

double RuntimeProfiler::simd_utilization() const
{
    return simdUtilization_;
}

//==============================================================

double RuntimeProfiler::cache_efficiency() const
{
    return cacheEfficiency_;
}

//==============================================================

double RuntimeProfiler::parallel_efficiency() const
{
    return parallelEfficiency_;
}

//==============================================================

double RuntimeProfiler::reasoning_cost() const
{
    return reasoningCost_;
}

//==============================================================

double RuntimeProfiler::prediction_cost() const
{
    return predictionCost_;
}

//==============================================================

double RuntimeProfiler::scheduling_cost() const
{
    return schedulingCost_;
}

}
}