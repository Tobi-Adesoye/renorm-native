#include "renorm/runtime/lock_free_queue.h"

#include <cassert>
#include <utility>

namespace renorm
{
namespace runtime
{

template<typename T>
std::size_t
LockFreeQueue<T>::next_power_of_two(
    std::size_t value)
{
    if (value < 2)
        return 2;

    --value;

    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;

#if SIZE_MAX > UINT32_MAX
    value |= value >> 32;
#endif

    return ++value;
}

//==============================================================

template<typename T>
LockFreeQueue<T>::LockFreeQueue(
    std::size_t capacity)
    :
    capacity_(next_power_of_two(capacity)),
    mask_(capacity_ - 1),
    enqueuePos_(0),
    dequeuePos_(0)
{
    buffer_ =
        static_cast<Cell*>(
            ::operator new[](
                sizeof(Cell) * capacity_));

    for (std::size_t i = 0;
         i < capacity_;
         ++i)
    {
        new (&buffer_[i]) Cell;

        buffer_[i].sequence.store(
            i,
            std::memory_order_relaxed);
    }
}

//==============================================================

template<typename T>
LockFreeQueue<T>::~LockFreeQueue()
{
    T value;

    while (pop(value))
    {
    }

    for (std::size_t i = 0;
         i < capacity_;
         ++i)
    {
        buffer_[i].~Cell();
    }

    ::operator delete[](buffer_);
}

//==============================================================

template<typename T>
T*
LockFreeQueue<T>::value_ptr(
    Cell& cell)
{
    return reinterpret_cast<T*>(
        &cell.storage);
}

template<typename T>
const T*
LockFreeQueue<T>::value_ptr(
    const Cell& cell) const
{
    return reinterpret_cast<const T*>(
        &cell.storage);
}

//==============================================================

template<typename T>
bool
LockFreeQueue<T>::push(
    const T& value)
{
    std::size_t pos =
        enqueuePos_.load(
            std::memory_order_relaxed);

    for (;;)
    {
        Cell& cell =
            buffer_[pos & mask_];

        std::size_t seq =
            cell.sequence.load(
                std::memory_order_acquire);

        intptr_t diff =
            static_cast<intptr_t>(seq) -
            static_cast<intptr_t>(pos);

        if (diff == 0)
        {
            if (enqueuePos_.compare_exchange_weak(
                    pos,
                    pos + 1,
                    std::memory_order_relaxed))
            {
                new (value_ptr(cell)) T(value);

                cell.sequence.store(
                    pos + 1,
                    std::memory_order_release);

                return true;
            }
        }
        else if (diff < 0)
        {
            return false;
        }
        else
        {
            pos =
                enqueuePos_.load(
                    std::memory_order_relaxed);
        }
    }
}

//==============================================================

template<typename T>
bool
LockFreeQueue<T>::push(
    T&& value)
{
    std::size_t pos =
        enqueuePos_.load(
            std::memory_order_relaxed);

    for (;;)
    {
        Cell& cell =
            buffer_[pos & mask_];

        std::size_t seq =
            cell.sequence.load(
                std::memory_order_acquire);

        intptr_t diff =
            static_cast<intptr_t>(seq) -
            static_cast<intptr_t>(pos);

        if (diff == 0)
        {
            if (enqueuePos_.compare_exchange_weak(
                    pos,
                    pos + 1,
                    std::memory_order_relaxed))
            {
                new (value_ptr(cell))
                    T(std::move(value));

                cell.sequence.store(
                    pos + 1,
                    std::memory_order_release);

                return true;
            }
        }
        else if (diff < 0)
        {
            return false;
        }
        else
        {
            pos =
                enqueuePos_.load(
                    std::memory_order_relaxed);
        }
    }
}

//==============================================================

template<typename T>
bool
LockFreeQueue<T>::pop(
    T& value)
{
    std::size_t pos =
        dequeuePos_.load(
            std::memory_order_relaxed);

    for (;;)
    {
        Cell& cell =
            buffer_[pos & mask_];

        std::size_t seq =
            cell.sequence.load(
                std::memory_order_acquire);

        intptr_t diff =
            static_cast<intptr_t>(seq) -
            static_cast<intptr_t>(pos + 1);

        if (diff == 0)
        {
            if (dequeuePos_.compare_exchange_weak(
                    pos,
                    pos + 1,
                    std::memory_order_relaxed))
            {
                T* ptr =
                    value_ptr(cell);

                value =
                    std::move(*ptr);

                ptr->~T();

                cell.sequence.store(
                    pos + capacity_,
                    std::memory_order_release);

                return true;
            }
        }
        else if (diff < 0)
        {
            return false;
        }
        else
        {
            pos =
                dequeuePos_.load(
                    std::memory_order_relaxed);
        }
    }
}

//==============================================================

template<typename T>
std::size_t
LockFreeQueue<T>::size() const noexcept
{
    std::size_t e =
        enqueuePos_.load(
            std::memory_order_acquire);

    std::size_t d =
        dequeuePos_.load(
            std::memory_order_acquire);

    return e - d;
}

//==============================================================

template<typename T>
std::size_t
LockFreeQueue<T>::capacity() const noexcept
{
    return capacity_;
}

template<typename T>
bool
LockFreeQueue<T>::empty() const noexcept
{
    return size() == 0;
}

} // namespace runtime
} // namespace renorm