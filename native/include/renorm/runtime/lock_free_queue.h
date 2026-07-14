#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// LockFreeQueue
//
// Bounded MPMC lock-free queue.
//
// Characteristics:
//
// • Multiple producers
// • Multiple consumers
// • Wait-free fast path
// • No mutexes
// • Cache-line aligned
//
//==============================================================

template<typename T>
class LockFreeQueue
{
public:

    explicit LockFreeQueue(std::size_t capacity);

    ~LockFreeQueue();

    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;

    //----------------------------------------------------------
    // Push
    //----------------------------------------------------------

    bool push(const T& value);

    bool push(T&& value);

    //----------------------------------------------------------
    // Pop
    //----------------------------------------------------------

    bool pop(T& value);

    //----------------------------------------------------------
    // Approximate size
    //----------------------------------------------------------

    std::size_t size() const noexcept;

    //----------------------------------------------------------
    // Capacity
    //----------------------------------------------------------

    std::size_t capacity() const noexcept;

    //----------------------------------------------------------
    // Empty
    //----------------------------------------------------------

    bool empty() const noexcept;

private:

    struct alignas(64) Cell
    {
        std::atomic<std::size_t> sequence;

        typename std::aligned_storage<
            sizeof(T),
            alignof(T)
        >::type storage;
    };

    Cell* buffer_;

    std::size_t capacity_;

    std::size_t mask_;

    alignas(64)
    std::atomic<std::size_t> enqueuePos_;

    alignas(64)
    std::atomic<std::size_t> dequeuePos_;

private:

    static std::size_t
    next_power_of_two(
        std::size_t value);

    T* value_ptr(Cell& cell);

    const T* value_ptr(
        const Cell& cell) const;
};

} // namespace runtime
} // namespace renorm

#include "lock_free_queue.inl"