#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace renorm
{

//==============================================================
// WorkspaceAllocator
//
// High-performance linear allocator used by GEMM workers.
//
// Characteristics:
//
//  • 64-byte aligned allocations
//  • O(1) allocation
//  • O(1) reset
//  • No frees
//  • Reused every GEMM invocation
//
// Layout:
//
// +----------------------------------------------------+
// |                reserved buffer                     |
// +----------------------------------------------------+
// ^
// base_
//
// offset_ always advances forward.
//
//==============================================================

class WorkspaceAllocator
{
public:

    //----------------------------------------------------------
    // Alignment
    //----------------------------------------------------------

    static constexpr std::size_t ALIGNMENT = 64;

public:

    WorkspaceAllocator() = default;

    //----------------------------------------------------------
    // Reserve Capacity
    //
    // Reserve enough storage for all future allocations.
    //
    //----------------------------------------------------------

    void reserve(std::size_t bytes);

    //----------------------------------------------------------
    // Reset allocator
    //----------------------------------------------------------

    void reset()
    {
        offset_ = 0;
    }

    //----------------------------------------------------------
    // Allocate
    //
    // Returns a 64-byte aligned pointer.
    //
    //----------------------------------------------------------

    template<typename T>
    T* allocate(std::size_t count)
    {
        const std::size_t bytes =
            sizeof(T) * count;

        uintptr_t current =
            reinterpret_cast<uintptr_t>(
                storage_.data()) + offset_;

        uintptr_t aligned =
            (current + (ALIGNMENT - 1)) &
            ~(uintptr_t)(ALIGNMENT - 1);

        std::size_t alignedOffset =
            aligned -
            reinterpret_cast<uintptr_t>(
                storage_.data());

        if (alignedOffset + bytes > storage_.size())
        {
            throw std::bad_alloc();
        }

        T* result =
            reinterpret_cast<T*>(
                storage_.data() + alignedOffset);

        offset_ =
            alignedOffset + bytes;

        return result;
    }

    //----------------------------------------------------------
    // Capacity
    //----------------------------------------------------------

    std::size_t capacity() const
    {
        return storage_.size();
    }

    //----------------------------------------------------------
    // Used bytes
    //----------------------------------------------------------

    std::size_t used() const
    {
        return offset_;
    }

    //----------------------------------------------------------
    // Remaining bytes
    //----------------------------------------------------------

    std::size_t remaining() const
    {
        return storage_.size() - offset_;
    }

private:

    //----------------------------------------------------------
    // Raw storage
    //
    // Extra alignment padding is reserved so every allocation
    // can begin on a cache-line boundary.
    //----------------------------------------------------------

    std::vector<std::uint8_t> storage_;

    //----------------------------------------------------------
    // Current bump offset
    //----------------------------------------------------------

    std::size_t offset_ = 0;
};

} // namespace renorm