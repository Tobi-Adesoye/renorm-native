#pragma once

#include <cstddef>
#include <cstdint>

#include "renorm/workspace_allocator.h"

namespace renorm {

class KernelContext
{
public:

    KernelContext();

    ~KernelContext();

    //==========================================================
    // Workspace
    //==========================================================

    WorkspaceAllocator& workspace();

    //==========================================================
    // Statistics
    //==========================================================

    std::uint64_t kernel_calls() const;

    void increment_kernel_calls();

    void reset_statistics();

private:

    WorkspaceAllocator workspace_;

    std::uint64_t kernel_calls_;
};

} // namespace renorm