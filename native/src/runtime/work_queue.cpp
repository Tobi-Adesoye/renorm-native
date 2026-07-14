#include "renorm/runtime/work_queue.h"

namespace renorm
{
namespace runtime
{

WorkQueue::WorkQueue()
{
}

//==============================================================

void WorkQueue::push(Task task)
{
    {
        std::lock_guard<std::mutex> lock(mMutex);

        if (mShutdown)
            return;

        mQueue.push(std::move(task));
    }

    mCondition.notify_one();
}

//==============================================================

bool WorkQueue::pop(Task& task)
{
    std::lock_guard<std::mutex> lock(mMutex);

    if (mQueue.empty())
        return false;

    task = std::move(mQueue.front());

    mQueue.pop();

    return true;
}

//==============================================================

bool WorkQueue::wait_pop(Task& task)
{
    std::unique_lock<std::mutex> lock(mMutex);

    mCondition.wait(
        lock,
        [&]()
        {
            return mShutdown || !mQueue.empty();
        });

    if (mShutdown)
        return false;

    task = std::move(mQueue.front());

    mQueue.pop();

    return true;
}

//==============================================================

void WorkQueue::shutdown()
{
    {
        std::lock_guard<std::mutex> lock(mMutex);

        mShutdown = true;
    }

    mCondition.notify_all();
}

//==============================================================

void WorkQueue::clear()
{
    std::lock_guard<std::mutex> lock(mMutex);

    while (!mQueue.empty())
        mQueue.pop();
}

//==============================================================

bool WorkQueue::empty() const
{
    std::lock_guard<std::mutex> lock(mMutex);

    return mQueue.empty();
}

//==============================================================

std::size_t WorkQueue::size() const
{
    std::lock_guard<std::mutex> lock(mMutex);

    return mQueue.size();
}

} // namespace runtime
} // namespace renorm