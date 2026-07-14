#include "renorm/runtime/thread_profiler.h"

#include "renorm/runtime/parallel_statistics.h"

namespace renorm
{
namespace runtime
{

//==============================================================

ThreadProfiler::ThreadProfiler(
    std::size_t workerCount)
{
    clocks_.resize(workerCount);

    profiles_.resize(workerCount);
}

//==============================================================

void ThreadProfiler::begin_execution(
    std::size_t workerId)
{
    clocks_[workerId].executionStart =
        Clock::now();
}

void ThreadProfiler::end_execution(
    std::size_t workerId)
{
    auto elapsed =
        Clock::now() -
        clocks_[workerId].executionStart;

    profiles_[workerId].executionTimeMs +=
        std::chrono::duration<double,
            std::milli>(elapsed).count();

    profiles_[workerId].activeTimeMs +=
        std::chrono::duration<double,
            std::milli>(elapsed).count();
}

//==============================================================

void ThreadProfiler::begin_idle(
    std::size_t workerId)
{
    clocks_[workerId].idleStart =
        Clock::now();
}

void ThreadProfiler::end_idle(
    std::size_t workerId)
{
    auto elapsed =
        Clock::now() -
        clocks_[workerId].idleStart;

    profiles_[workerId].idleTimeMs +=
        std::chrono::duration<double,
            std::milli>(elapsed).count();
}

//==============================================================

void ThreadProfiler::begin_sync(
    std::size_t workerId)
{
    clocks_[workerId].syncStart =
        Clock::now();
}

void ThreadProfiler::end_sync(
    std::size_t workerId)
{
    auto elapsed =
        Clock::now() -
        clocks_[workerId].syncStart;

    profiles_[workerId].synchronizationTimeMs +=
        std::chrono::duration<double,
            std::milli>(elapsed).count();
}

//==============================================================

void ThreadProfiler::finalize(
    const ParallelStatistics&)
{
    for (auto& profile : profiles_)
    {
        const double total =
            profile.activeTimeMs +
            profile.idleTimeMs;

        if (total > 0.0)
        {
            profile.utilization =
                profile.activeTimeMs /
                total;
        }

        const double work =
            profile.executionTimeMs;

        const double overhead =
            profile.idleTimeMs +
            profile.synchronizationTimeMs;

        if ((work + overhead) > 0.0)
        {
            profile.schedulingEfficiency =
                work /
                (work + overhead);
        }
    }
}

//==============================================================

const ThreadProfiler::ThreadProfile&
ThreadProfiler::profile(
    std::size_t workerId) const
{
    return profiles_[workerId];
}

//==============================================================

double
ThreadProfiler::average_utilization() const
{
    double total = 0.0;

    for (const auto& profile : profiles_)
    {
        total += profile.utilization;
    }

    return profiles_.empty()
        ? 0.0
        : total / profiles_.size();
}

//==============================================================

double
ThreadProfiler::average_efficiency() const
{
    double total = 0.0;

    for (const auto& profile : profiles_)
    {
        total +=
            profile.schedulingEfficiency;
    }

    return profiles_.empty()
        ? 0.0
        : total / profiles_.size();
}

} // namespace runtime
} // namespace renorm