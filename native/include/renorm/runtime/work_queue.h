#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace renorm
{
namespace runtime
{

//==============================================================
// WorkQueue
//
// Thread-safe task queue.
//
// RuntimeExecutionEngine
//          ↓
//     ParallelRuntime
//          ↓
//       WorkQueue
//          ↓
//     WorkerThread(s)
//==============================================================

class WorkQueue
{
public:

    using Task = std::function<void()>;

    WorkQueue();

    ~WorkQueue() = default;

    //----------------------------------------------------------
    // Queue operations
    //----------------------------------------------------------

    void push(Task task);

    bool pop(Task& task);

    bool wait_pop(Task& task);

    //----------------------------------------------------------
    // Control
    //----------------------------------------------------------

    void shutdown();

    void clear();

    //----------------------------------------------------------
    // Status
    //----------------------------------------------------------

    bool empty() const;

    std::size_t size() const;

private:

    mutable std::mutex mMutex;

    std::condition_variable mCondition;

    std::queue<Task> mQueue;

    std::atomic<bool> mShutdown{false};
};

}
}