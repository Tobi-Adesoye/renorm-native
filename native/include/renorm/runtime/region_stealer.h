#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// RegionStealer
//
// Dynamic work distribution.
//
// Each worker owns a deque.
//
// Workers:
//
//   • pop_front() locally
//   • steal_back() remotely
//
// Foundation for:
//
//   • Parallel frontier expansion
//   • NUMA scheduling
//   • CPU affinity
//
//==============================================================

class RegionStealer
{
public:

    explicit RegionStealer(
        std::size_t workerCount);

    //----------------------------------------------------------
    // Initialize work
    //----------------------------------------------------------

    void initialize(
        const std::vector<std::size_t>& regions);

    //----------------------------------------------------------
    // Local work
    //----------------------------------------------------------

    std::optional<std::size_t>
    pop_local(
        std::size_t workerId);

    //----------------------------------------------------------
    // Remote stealing
    //----------------------------------------------------------

    std::optional<std::size_t>
    steal(
        std::size_t workerId);

    //----------------------------------------------------------
    // Finished?
    //----------------------------------------------------------

    bool empty() const;

private:

    struct WorkerQueue
    {
        std::deque<std::size_t> queue;

        mutable std::mutex mutex;
    };

    std::vector<WorkerQueue> workers_;

    std::atomic<std::size_t> remaining_{0};
};

} // namespace runtime
} // namespace renorm