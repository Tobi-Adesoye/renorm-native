#pragma once

#include <cstddef>
#include <vector>

namespace renorm
{
namespace runtime
{

class ExecutionContext;
template<typename T>
class LockFreeQueue;

//==============================================================
//
// AdaptiveScheduler
//
// Dynamically distributes work across worker queues.
//
// Responsibilities:
//
// • Initial work distribution
// • Dynamic load balancing
// • Queue pressure monitoring
// • Scheduling heuristics
//
//==============================================================

class AdaptiveScheduler
{
public:

    explicit AdaptiveScheduler(
        std::size_t workerCount);

    //----------------------------------------------------------
    // Prepare scheduling state
    //----------------------------------------------------------

    void initialize();

    //----------------------------------------------------------
    // Schedule regions
    //----------------------------------------------------------

    void schedule(
        const std::vector<std::size_t>& regions);

    //----------------------------------------------------------
    // Pop work for worker
    //----------------------------------------------------------

    bool pop(
        std::size_t workerId,
        std::size_t& region);

    //----------------------------------------------------------
    // Push work
    //----------------------------------------------------------

    bool push(
        std::size_t workerId,
        std::size_t region);

    //----------------------------------------------------------
    // Queue size
    //----------------------------------------------------------

    std::size_t queue_size(
        std::size_t workerId) const;

private:

    std::size_t least_loaded_worker() const;

private:

    std::size_t workerCount_;

    std::vector<
        LockFreeQueue<std::size_t>> queues_;
};

}
}