#include "renorm/runtime/thread_pool.h"

namespace renorm
{
namespace runtime
{

//==============================================================

ThreadPool::ThreadPool(
    std::size_t threadCount)
{
    if (threadCount == 0)
        threadCount = 1;

    workers_.reserve(threadCount);

    for (std::size_t i = 0;
         i < threadCount;
         ++i)
    {
        workers_.emplace_back(
            &ThreadPool::worker,
            this);
    }
}

//==============================================================

ThreadPool::~ThreadPool()
{
    shutdown();
}

//==============================================================

void ThreadPool::shutdown()
{
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        terminate_ = true;
    }

    cv_.notify_all();

    for (auto& worker :
         workers_)
    {
        if (worker.joinable())
            worker.join();
    }

    workers_.clear();
}

//==============================================================

std::size_t
ThreadPool::size() const noexcept
{
    return workers_.size();
}

//==============================================================

void ThreadPool::worker()
{
    while (true)
    {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(
                mutex_);

            cv_.wait(
                lock,
                [this]
                {
                    return terminate_
                        || !tasks_.empty();
                });

            if (terminate_ &&
                tasks_.empty())
            {
                return;
            }

            task =
                std::move(
                    tasks_.front());

            tasks_.pop();
        }

        task();
    }
}

} // runtime
} // renorm