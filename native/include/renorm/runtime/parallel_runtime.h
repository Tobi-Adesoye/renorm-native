#pragma once

#include "renorm/runtime/execution_context.h"
#include "renorm/runtime/worker_thread.h"

#include <vector>
#include <memory>

namespace renorm
{
namespace runtime
{

//==============================================================
// ParallelRuntime
//
// Owns the worker pool used by the runtime.
//
// RuntimeExecutionEngine
//          │
//          ▼
//     ParallelRuntime
//          │
//          ▼
//     Worker Threads
//==============================================================

class ParallelRuntime
{
public:

    ParallelRuntime();

    explicit ParallelRuntime(
        std::size_t workerCount);

    ~ParallelRuntime();

    //----------------------------------------------------------
    // Lifecycle
    //----------------------------------------------------------

    void initialize();

    void shutdown();

    //----------------------------------------------------------
    // Execute one runtime iteration
    //----------------------------------------------------------

    void execute(
        ExecutionContext& context);

    //----------------------------------------------------------
    // Worker information
    //----------------------------------------------------------

    std::size_t worker_count() const;

    void resize(
        std::size_t workers);

private:

    //----------------------------------------------------------
    // Execute all scheduled regions
    //----------------------------------------------------------

    void dispatch_regions(
        ExecutionContext& context);

    //----------------------------------------------------------
    // Wait for completion
    //----------------------------------------------------------

    void synchronize();

private:

    std::vector<std::unique_ptr<WorkerThread>> workers;

    std::size_t workerCount = 0;
};

}
}