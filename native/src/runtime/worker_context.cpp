#include "renorm/worker_context.h"

namespace renorm
{

//==============================================================
// Construction
//==============================================================

WorkerContext::WorkerContext()
{
    //----------------------------------------------------------
    // Reserve workspace once.
    //
    // The buffer is reused for every GEMM invocation executed
    // by this worker, eliminating repeated heap allocations.
    //----------------------------------------------------------

    workspace.reserve(DefaultWorkspace);
}

//==============================================================
// Reset Workspace
//==============================================================

void WorkerContext::reset()
{
    //----------------------------------------------------------
    // Rewind allocation pointer.
    //
    // Memory is not freed.
    // Previously allocated storage is reused.
    //----------------------------------------------------------

    workspace.reset();
}

} // namespace renorm