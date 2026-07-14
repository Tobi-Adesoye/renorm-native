#pragma once

#include <deque>
#include <mutex>
#include <optional>

namespace renorm
{

template <typename T>
class WorkStealingQueue
{
public:

    void push(T value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        deque_.push_back(std::move(value));
    }

    std::optional<T> pop()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (deque_.empty())
            return std::nullopt;

        T value = std::move(deque_.back());
        deque_.pop_back();
        return value;
    }

    std::optional<T> steal()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (deque_.empty())
            return std::nullopt;

        T value = std::move(deque_.front());
        deque_.pop_front();
        return value;
    }

private:

    std::deque<T> deque_;
    std::mutex mutex_;
};

} // namespace renorm