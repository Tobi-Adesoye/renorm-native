#include "renorm/runtime/runtime_barrier.h"

namespace renorm
{
namespace runtime
{

//==============================================================

RuntimeBarrier::RuntimeBarrier(
    std::size_t participants)
    :
    participants_(participants),
    waiting_(0),
    generation_(0)
{
}

//==============================================================

void RuntimeBarrier::wait()
{
    std::unique_lock<std::mutex> lock(
        mutex_);

    const std::size_t currentGeneration =
        generation_.load(
            std::memory_order_relaxed);

    ++waiting_;

    //----------------------------------------------------------
    // Last thread releases everyone
    //----------------------------------------------------------

    if (waiting_ == participants_)
    {
        waiting_ = 0;

        generation_.fetch_add(
            1,
            std::memory_order_release);

        lock.unlock();

        condition_.notify_all();

        return;
    }

    //----------------------------------------------------------
    // Wait for next generation
    //----------------------------------------------------------

    condition_.wait(
        lock,
        [&]()
        {
            return
                generation_.load(
                    std::memory_order_acquire)
                != currentGeneration;
        });
}

//==============================================================

void RuntimeBarrier::reset(
    std::size_t participants)
{
    std::lock_guard<std::mutex> lock(
        mutex_);

    participants_ = participants;

    waiting_ = 0;

    generation_.store(
        0,
        std::memory_order_release);
}

//==============================================================

std::size_t
RuntimeBarrier::participants() const noexcept
{
    return participants_;
}

//==============================================================

std::size_t
RuntimeBarrier::generation() const noexcept
{
    return generation_.load(
        std::memory_order_acquire);
}

} // namespace runtime
} // namespace renorm