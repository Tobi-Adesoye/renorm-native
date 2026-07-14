#pragma once

#include <cstddef>

namespace renorm
{
namespace runtime
{

class ExecutionContext;

//==============================================================
//
// RuntimeOptimizer
//
// Dynamic execution optimizer.
//
// Executes after PredictionEngine and before AdaptivePruner.
//
// Responsibilities
//
// • Dynamic block fusion
// • Branch elimination
// • SIMD specialization
// • Region specialization
// • Cache optimization
// • Runtime tuning
//
//==============================================================

class RuntimeOptimizer
{
public:

    RuntimeOptimizer() = default;

    //----------------------------------------------------------
    // Optimize runtime execution
    //----------------------------------------------------------

    void optimize(
        ExecutionContext& context) const;

private:

    //----------------------------------------------------------
    // Optimization stages
    //----------------------------------------------------------

    void optimize_blocks(
        ExecutionContext& context) const;

    void optimize_regions(
        ExecutionContext& context) const;

    void optimize_branches(
        ExecutionContext& context) const;

    void optimize_simd(
        ExecutionContext& context) const;

    void optimize_cache(
        ExecutionContext& context) const;

    void optimize_memory(
        ExecutionContext& context) const;

    void optimize_parallelism(
        ExecutionContext& context) const;

    //----------------------------------------------------------
    // Runtime metrics
    //----------------------------------------------------------

    float compute_execution_efficiency(
        const ExecutionContext& context) const;

    float compute_cache_efficiency(
        const ExecutionContext& context) const;

    float compute_parallel_efficiency(
        const ExecutionContext& context) const;
};

}
}