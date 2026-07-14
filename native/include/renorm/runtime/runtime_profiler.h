#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <unordered_map>

namespace renorm
{
namespace runtime
{

class ExecutionContext;

//==============================================================
//
// RuntimeProfiler
//
// Global runtime instrumentation layer.
//
// Measures:
//
// • Compilation
// • Region scheduling
// • SIMD execution
// • Reasoning
// • Prediction
// • Optimization
// • Runtime specialization
// • Memory
// • Parallel execution
//
//==============================================================

class RuntimeProfiler
{
public:

    enum class Stage
    {
        Compilation,

        BlockCompilation,

        RegionCompilation,

        Scheduling,

        RuntimeOptimization,

        RuntimeSpecialization,

        SIMDExecution,

        Prediction,

        Reasoning,

        AdaptivePruning,

        ParallelExecution,

        Memory,

        Cache,

        Finalization
    };

public:

    RuntimeProfiler();

    //----------------------------------------------------------
    // Stage Timing
    //----------------------------------------------------------

    void begin(Stage stage);

    void end(Stage stage);

    //----------------------------------------------------------
    // Runtime
    //----------------------------------------------------------

    void update(
        const ExecutionContext& context);

    //----------------------------------------------------------
    // Reset
    //----------------------------------------------------------

    void reset();

    //----------------------------------------------------------
    // Queries
    //----------------------------------------------------------

    double total_runtime_ms() const;

    double stage_runtime_ms(Stage stage) const;

    double simd_utilization() const;

    double cache_efficiency() const;

    double parallel_efficiency() const;

    double reasoning_cost() const;

    double prediction_cost() const;

    double scheduling_cost() const;

private:

    using Clock =
        std::chrono::steady_clock;

    struct StageTimer
    {
        Clock::time_point start;

        double elapsedMs = 0.0;
    };

private:

    std::unordered_map<Stage, StageTimer> timers_;

    double simdUtilization_ = 0.0;

    double cacheEfficiency_ = 0.0;

    double parallelEfficiency_ = 0.0;

    double reasoningCost_ = 0.0;

    double predictionCost_ = 0.0;

    double schedulingCost_ = 0.0;
};

}
}