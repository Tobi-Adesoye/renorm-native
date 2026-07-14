#pragma once

#include <memory>

namespace renorm
{
namespace runtime
{

// Forward declaration to minimize header coupling
struct ExecutionContext;

//==============================================================
// RuntimeExecutionEngine
//
// Central runtime execution pipeline.
//
// Pipeline:
//
//   ExecutionContext
//          ↓
//   Region Scheduler
//          ↓
//   Block Executor
//          ↓
//   Prediction Runtime
//          ↓
//   Reasoning Engine
//          ↓
//   Runtime Optimizer
//          ↓
//   Adaptive Pruner
//          ↓
//   RuntimeResultBuilder
//          ↓
//   RuntimeResult
//==============================================================

class RuntimeExecutionEngine
{
public:

    RuntimeExecutionEngine();
    ~RuntimeExecutionEngine();

    // Explicit Rule of Five layout
    RuntimeExecutionEngine(const RuntimeExecutionEngine&) = delete;
    RuntimeExecutionEngine& operator=(const RuntimeExecutionEngine&) = delete;
    
    RuntimeExecutionEngine(RuntimeExecutionEngine&&) noexcept;
    RuntimeExecutionEngine& operator=(RuntimeExecutionEngine&&) noexcept;

    //----------------------------------------------------------
    // Execute entire runtime
    //----------------------------------------------------------

    void execute(ExecutionContext& context) const;

private:

    //----------------------------------------------------------
    // Pipeline stages
    //----------------------------------------------------------

    void initialize(ExecutionContext& context) const;

    void schedule_regions(ExecutionContext& context) const;

    void execute_blocks(ExecutionContext& context) const;

    void execute_prediction(ExecutionContext& context) const;

    void execute_reasoning(ExecutionContext& context) const;

    void optimize(ExecutionContext& context) const;

    void prune(ExecutionContext& context) const;

    void finalize(ExecutionContext& context) const;

    //----------------------------------------------------------
    // Core loop policies
    //----------------------------------------------------------

    bool should_terminate(const ExecutionContext& context) const;

private:

    //----------------------------------------------------------
    // Opaque Implementation (PIMPL)
    //----------------------------------------------------------

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace renorm