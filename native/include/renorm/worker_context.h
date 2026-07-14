#pragma once

#include "renorm/workspace_allocator.h"

namespace renorm
{

//==============================================================
// WorkerContext
//
// Per-thread execution context.
//
// Each worker owns:
//
//  • A reusable aligned workspace
//  • Future scratch buffers
//  • Future statistics
//
// One WorkerContext exists per thread and is reused across
// GEMM invocations to eliminate repeated allocations.
//
//==============================================================

class WorkerContext
{
public:

    WorkerContext() = default;

    ~WorkerContext() = default;

    WorkerContext(const WorkerContext&) = delete;
    WorkerContext& operator=(const WorkerContext&) = delete;

    WorkerContext(WorkerContext&&) = default;
    WorkerContext& operator=(WorkerContext&&) = default;

    //----------------------------------------------------------
    // Aligned Workspace
    //----------------------------------------------------------

    WorkspaceAllocator workspace;
};

} // namespace renorm