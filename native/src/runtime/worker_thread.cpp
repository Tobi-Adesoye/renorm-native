#include "renorm/runtime/worker_thread.h"

namespace renorm
{
namespace runtime
{

WorkerThread::WorkerThread()
{
}

WorkerThread::~WorkerThread()
{
    stop();
}

//==============================================================

void WorkerThread::start()
{
    if (mRunning)
        return;

    mRunning = true;

    mThread =
        std::thread(
            &WorkerThread::thread_loop,
            this);
}

//==============================================================

void WorkerThread::stop()
{
    if (!mRunning)
        return;

    {
        std::lock_guard<std::mutex> lock(mMutex);

        mRunning = false;

        mHasJob = true;
    }

    mCondition.notify_one();

    if (mThread.joinable())
        mThread.join();
}

//==============================================================

void WorkerThread::execute(
    std::function<void()> job)
{
    {
        std::lock_guard<std::mutex> lock(mMutex);

        mJob = std::move(job);

        mHasJob = true;

        mBusy = true;
    }

    mCondition.notify_one();
}

//==============================================================

void WorkerThread::wait()
{
    std::unique_lock<std::mutex> lock(mMutex);

    mFinished.wait(
        lock,
        [&]
        {
            return !mBusy.load();
        });
}

//==============================================================

bool WorkerThread::busy() const
{
    return mBusy;
}

//==============================================================

void WorkerThread::thread_loop()
{
    while (true)
    {
        std::function<void()> job;

        {
            std::unique_lock<std::mutex> lock(mMutex);

            mCondition.wait(
                lock,
                [&]
                {
                    return mHasJob;
                });

            if (!mRunning)
                break;

            job = std::move(mJob);

            mHasJob = false;
        }

        //------------------------------------------------------
        // Execute
        //------------------------------------------------------

        if (job)
            job();

        //------------------------------------------------------
        // Finished
        //------------------------------------------------------

        {
            std::lock_guard<std::mutex> lock(mMutex);

            mBusy = false;
        }

        mFinished.notify_all();
    }

    mBusy = false;

    mFinished.notify_all();
}

}
}