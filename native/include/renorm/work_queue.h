#pragma once

#include <deque>
#include <functional>
#include <mutex>
#include <optional>

namespace renorm
{

struct WorkerContext;

template<typename T>
class WorkStealingDeque
{
public:

    void push(T item)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        deque_.push_back(std::move(item));
    }

    std::optional<T> pop()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (deque_.empty())
            return std::nullopt;

        T item = std::move(deque_.back());
        deque_.pop_back();

        return item;
    }

    std::optional<T> steal()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (deque_.empty())
            return std::nullopt;

        T item = std::move(deque_.front());
        deque_.pop_front();

        return item;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return deque_.empty();
    }

private:

    std::deque<T> deque_;

    mutable std::mutex mutex_;
};

using Job = std::function<void(WorkerContext&)>;

} // namespace renorm