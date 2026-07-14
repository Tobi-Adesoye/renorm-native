#pragma once

#include <memory>

namespace renorm
{
namespace runtime
{

class ExecutionContext;

//==============================================================
//
// RuntimeOrchestrator
//
// Top-level runtime coordinator.
//
// Pipeline
//
// Graph
//      ↓
// Compiler
//      ↓
// Compiled Blocks
//      ↓
// Compiled Regions
//      ↓
// Runtime Optimizer
//      ↓
// Runtime Specializer
//      ↓
// Execution Plan Optimizer
//      ↓
// Cache Optimizer
//      ↓
// Memory Prefetcher
//      ↓
// Parallel Scheduler
//      ↓
// SIMD Executor
//      ↓
// Reasoning Engine
//      ↓
// Adaptive Pruner
//      ↓
// Runtime Profiler
//      ↓
// Runtime Result
//
//==============================================================

class RuntimeOrchestrator
{
public:

    RuntimeOrchestrator();
    ~RuntimeOrchestrator();

    RuntimeOrchestrator(
        const RuntimeOrchestrator&) = delete;

    RuntimeOrchestrator&
    operator=(
        const RuntimeOrchestrator&) = delete;

    RuntimeOrchestrator(
        RuntimeOrchestrator&&) noexcept;

    RuntimeOrchestrator&
    operator=(
        RuntimeOrchestrator&&) noexcept;

    //----------------------------------------------------------

    void execute(
        ExecutionContext& context);

private:

    void initialize(
        ExecutionContext& context);

    void optimize(
        ExecutionContext& context);

    void specialize(
        ExecutionContext& context);

    void build_execution_plan(
        ExecutionContext& context);

    void prepare_runtime(
        ExecutionContext& context);

    void execute_runtime(
        ExecutionContext& context);

    void reasoning(
        ExecutionContext& context);

    void prune(
        ExecutionContext& context);

    void finalize(
        ExecutionContext& context);

private:

    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}
}