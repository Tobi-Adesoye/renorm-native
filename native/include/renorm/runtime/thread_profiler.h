#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

namespace renorm
{
namespace runtime
{

class ParallelStatistics;

//==============================================================
//
// ThreadProfiler
//
// Collects per-thread runtime profiling information.
//
// Tracks:
//
// • Execution time
// • Active time
// • Idle time
// • Synchronization time
// • CPU utilization
// • Scheduling efficiency
//
//==============================================================

class ThreadProfiler
{
public:

    struct ThreadProfile
    {
        double executionTimeMs = 0.0;

        double activeTimeMs = 0.0;

        double idleTimeMs = 0.0;

        double synchronizationTimeMs = 0.0;

        double utilization = 0.0;

        double schedulingEfficiency = 0.0;
    };

public:

    explicit ThreadProfiler(
        std::size_t workerCount);

    //----------------------------------------------------------
    // Profiling lifecycle
    //----------------------------------------------------------

    void begin_execution(
        std::size_t workerId);

    void end_execution(
        std::size_t workerId);

    void begin_idle(
        std::size_t workerId);

    void end_idle(
        std::size_t workerId);

    void begin_sync(
        std::size_t workerId);

    void end_sync(
        std::size_t workerId);

    //----------------------------------------------------------
    // Finalize metrics
    //----------------------------------------------------------

    void finalize(
        const ParallelStatistics& statistics);

    //----------------------------------------------------------
    // Access
    //----------------------------------------------------------

    const ThreadProfile&
    profile(
        std::size_t workerId) const;

    double average_utilization() const;

    double average_efficiency() const;

private:

    using Clock =
        std::chrono::steady_clock;

    struct RuntimeClock
    {
        Clock::time_point executionStart;

        Clock::time_point idleStart;

        Clock::time_point syncStart;
    };

private:

    std::vector<RuntimeClock> clocks_;

    std::vector<ThreadProfile> profiles_;
};

}
}