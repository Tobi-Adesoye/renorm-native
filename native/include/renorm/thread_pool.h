#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// ThreadPool
//
// Fixed-size worker pool.
//
// Foundation for:
//
//   • SIMDRegionExecutor
//   • Region Stealing
//   • Parallel Frontier Expansion
//   • NUMA Scheduling
//
//==============================================================

class ThreadPool
{
public:

    explicit ThreadPool(
        std::size_t threadCount =
            std::thread::hardware_concurrency());

    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    //----------------------------------------------------------
    // Submit task
    //----------------------------------------------------------

    template<class F>
    auto submit(F&& task)
        -> std::future<
            typename std::invoke_result<F>::type>;

    //----------------------------------------------------------
    // Stop workers
    //----------------------------------------------------------

    void shutdown();

    //----------------------------------------------------------
    // Thread count
    //----------------------------------------------------------

    std::size_t size() const noexcept;

private:

    void worker();

private:

    std::vector<std::thread> workers_;

    std::queue<std::function<void()>> tasks_;

    mutable std::mutex mutex_;

    std::condition_variable cv_;

    bool terminate_ = false;
};

//==============================================================
// Template Implementation
//==============================================================

template<class F>
auto ThreadPool::submit(F&& task)
    -> std::future<
        typename std::invoke_result<F>::type>
{
    using ReturnType =
        typename std::invoke_result<F>::type;

    auto packaged =
        std::make_shared<
            std::packaged_task<ReturnType()>>(
                std::forward<F>(task));

    std::future<ReturnType> future =
        packaged->get_future();

    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        tasks_.emplace(
            [packaged]()
            {
                (*packaged)();
            });
    }

    cv_.notify_one();

    return future;
}

} // runtime
} // renorm