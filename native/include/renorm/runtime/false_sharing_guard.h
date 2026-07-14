#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// FalseSharingGuard
//
// Eliminates cache-line contention between worker threads.
//
// Modern x86 cache line:
//
//      64 bytes
//
// Every worker receives its own cache line.
//
//==============================================================

class FalseSharingGuard
{
public:

    static constexpr std::size_t CACHE_LINE_SIZE = 64;

    //----------------------------------------------------------
    // Aligned wrapper
    //----------------------------------------------------------

    template<typename T>
    struct alignas(CACHE_LINE_SIZE) Aligned
    {
        T value;

        char padding[
            CACHE_LINE_SIZE >
            sizeof(T)
                ? CACHE_LINE_SIZE -
                  sizeof(T)
                : 1];
    };

    //----------------------------------------------------------
    // Allocate aligned array
    //----------------------------------------------------------

    template<typename T>
    static Aligned<T>* allocate(
        std::size_t count)
    {
        return static_cast<
            Aligned<T>*>(

            ::operator new[](
                sizeof(Aligned<T>) *
                count,
                std::align_val_t(
                    CACHE_LINE_SIZE)));
    }

    //----------------------------------------------------------
    // Destroy aligned array
    //----------------------------------------------------------

    template<typename T>
    static void destroy(
        Aligned<T>* data)
    {
        ::operator delete[](
            data,
            std::align_val_t(
                CACHE_LINE_SIZE));
    }

    //----------------------------------------------------------
    // Cache line alignment
    //----------------------------------------------------------

    static constexpr bool
    is_cache_aligned(
        const void* ptr)
    {
        return
            (reinterpret_cast<
                std::uintptr_t>(ptr)
             &
             (CACHE_LINE_SIZE - 1))
            == 0;
    }
};

}
}