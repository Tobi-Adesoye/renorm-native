#pragma once

#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace renorm
{

template<typename T>
class WorkStealingDeque
{
public:

    WorkStealingDeque() = default;

    WorkStealingDeque(const WorkStealingDeque&) = delete;
    WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;

    //----------------------------------------------------------
    // Owner pushes work
    //----------------------------------------------------------

    void push(T item)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        deque_.push_back(std::move(item));
    }

    //----------------------------------------------------------
    // Owner pops from back (LIFO)
    //----------------------------------------------------------

    std::optional<T> pop()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (deque_.empty())
            return std::nullopt;

        T item = std::move(deque_.back());

        deque_.pop_back();

        return item;
    }

    //----------------------------------------------------------
    // Other workers steal from front (FIFO)
    //----------------------------------------------------------

    std::optional<T> steal()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (deque_.empty())
            return std::nullopt;

        T item = std::move(deque_.front());

        deque_.pop_front();

        return item;
    }

    //----------------------------------------------------------

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        return deque_.empty();
    }

    //----------------------------------------------------------

    std::size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        return deque_.size();
    }

private:

    mutable std::mutex mutex_;

    std::deque<T> deque_;
};

} // namespace renorm