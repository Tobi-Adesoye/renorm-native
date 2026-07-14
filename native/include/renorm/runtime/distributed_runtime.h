#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace renorm
{
namespace runtime
{

class ExecutionContext;

//==============================================================
//
// DistributedRuntime
//
// Coordinates execution across multiple workers.
//
// Responsibilities:
//
// • Partition work
// • Dispatch execution
// • Synchronize results
// • Handle worker lifecycle
// • Aggregate runtime metrics
//
//==============================================================

class DistributedRuntime
{
public:

    struct WorkerInfo
    {
        std::size_t id = 0;
        std::string endpoint;
        bool online = false;
    };

public:

    DistributedRuntime() = default;
    ~DistributedRuntime() = default;

    //----------------------------------------------------------

    void execute(
        ExecutionContext& context);

    //----------------------------------------------------------

    void add_worker(
        const WorkerInfo& worker);

    //----------------------------------------------------------

    std::size_t worker_count() const;

private:

    void partition(
        ExecutionContext& context);

    void dispatch(
        ExecutionContext& context);

    void synchronize(
        ExecutionContext& context);

    void aggregate(
        ExecutionContext& context);

private:

    std::vector<WorkerInfo> workers_;
};

} // namespace runtime
} // namespace renorm