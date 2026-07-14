#pragma once

#include <cstddef>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// CPUAffinityManager
//
// Pins worker threads to logical processors.
//
// Current
//
//  • Portable no-op implementation
//
// Future
//
//  • Linux
//      pthread_setaffinity_np
//
//  • Windows
//      SetThreadAffinityMask
//
//==============================================================

class CPUAffinityManager
{
public:

    CPUAffinityManager() = default;

    //----------------------------------------------------------
    // Pin current thread
    //----------------------------------------------------------

    bool pin_worker(
        std::size_t workerId) const;

    //----------------------------------------------------------
    // Remove affinity
    //----------------------------------------------------------

    bool unpin_current_thread() const;
};

}
}