#include "renorm/runtime/cpu_affinity_manager.h"

#ifdef _WIN32

#include <windows.h>

#elif defined(__linux__)

#include <pthread.h>
#include <sched.h>

#endif

namespace renorm
{
namespace runtime
{

//==============================================================

bool CPUAffinityManager::pin_worker(
    std::size_t workerId) const
{
#ifdef __linux__

    cpu_set_t mask;

    CPU_ZERO(&mask);

    CPU_SET(
        static_cast<int>(workerId),
        &mask);

    return
        pthread_setaffinity_np(

            pthread_self(),

            sizeof(mask),

            &mask) == 0;

#elif defined(_WIN32)

    DWORD_PTR mask =
        static_cast<DWORD_PTR>(1ull)
        << workerId;

    return
        SetThreadAffinityMask(

            GetCurrentThread(),

            mask) != 0;

#else

    (void)workerId;

    //----------------------------------------------------------
    // Portable fallback
    //----------------------------------------------------------

    return true;

#endif
}

//==============================================================

bool CPUAffinityManager::unpin_current_thread() const
{
#ifdef __linux__

    cpu_set_t mask;

    CPU_ZERO(&mask);

    const std::size_t cpuCount =
        static_cast<std::size_t>(
            CPU_SETSIZE);

    for (std::size_t i = 0;
         i < cpuCount;
         ++i)
    {
        CPU_SET(
            static_cast<int>(i),
            &mask);
    }

    return
        pthread_setaffinity_np(

            pthread_self(),

            sizeof(mask),

            &mask) == 0;

#elif defined(_WIN32)

    return
        SetThreadAffinityMask(

            GetCurrentThread(),

            ~static_cast<DWORD_PTR>(0)) != 0;

#else

    return true;

#endif
}

} // namespace runtime
} // namespace renorm