#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// RuntimeBarrier
//
// Reusable thread synchronization barrier.
//
// Characteristics
//
// • Reusable
// • Phase-based
// • Low overhead
// • Thread-safe
//
//==============================================================

class RuntimeBarrier
{
public:

    explicit RuntimeBarrier(
        std::size_t participants);

    RuntimeBarrier(
        const RuntimeBarrier&) = delete;

    RuntimeBarrier& operator=(
        const RuntimeBarrier&) = delete;

    //----------------------------------------------------------
    // Wait until all workers arrive
    //----------------------------------------------------------

    void wait();

    //----------------------------------------------------------
    // Reset barrier
    //----------------------------------------------------------

    void reset(
        std::size_t participants);

    //----------------------------------------------------------
    // Number of participants
    //----------------------------------------------------------

    std::size_t participants() const noexcept;

    //----------------------------------------------------------
    // Current generation
    //----------------------------------------------------------

    std::size_t generation() const noexcept;

private:

    std::mutex mutex_;

    std::condition_variable condition_;

    std::size_t participants_;

    std::size_t waiting_;

    std::atomic<std::size_t> generation_;
};

} // namespace runtime
} // namespace renorm