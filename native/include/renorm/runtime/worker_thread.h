#pragma once

#include <thread>
#include <atomic>
#include <functional>
#include <condition_variable>
#include <mutex>

namespace renorm
{
namespace runtime
{

//==============================================================
// WorkerThread
//
// Single reusable worker.
//
// RuntimeExecutionEngine owns several of these.
//
// A worker sleeps until work is assigned.
//==============================================================

class WorkerThread
{
public:

    WorkerThread();

    ~WorkerThread();

    //----------------------------------------------------------
    // Lifecycle
    //----------------------------------------------------------

    void start();

    void stop();

    //----------------------------------------------------------
    // Execute one job
    //----------------------------------------------------------

    void execute(
        std::function<void()> job);

    //----------------------------------------------------------
    // Synchronization
    //----------------------------------------------------------

    void wait();

    //----------------------------------------------------------
    // State
    //----------------------------------------------------------

    bool busy() const;

private:

    //----------------------------------------------------------
    // Main loop
    //----------------------------------------------------------

    void thread_loop();

    //----------------------------------------------------------
    // Thread
    //----------------------------------------------------------

    std::thread mThread;

    //----------------------------------------------------------
    // Synchronization
    //----------------------------------------------------------

    std::mutex mMutex;

    std::condition_variable mCondition;

    std::condition_variable mFinished;

    //----------------------------------------------------------
    // Job
    //----------------------------------------------------------

    std::function<void()> mJob;

    //----------------------------------------------------------
    // State
    //----------------------------------------------------------

    std::atomic<bool> mRunning{false};

    std::atomic<bool> mBusy{false};

    bool mHasJob = false;
};

}
}