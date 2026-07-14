#include "renorm/workspace_allocator.h"

#include <algorithm>

namespace renorm
{

//==============================================================
// Reserve Workspace
//
// The allocator reserves additional padding so that the first
// allocation (and every subsequent allocation after alignment)
// can begin on a 64-byte boundary.
//
//==============================================================

void WorkspaceAllocator::reserve(std::size_t bytes)
{
    //----------------------------------------------------------
    // Already large enough
    //----------------------------------------------------------

    if (storage_.size() >= bytes + ALIGNMENT)
    {
        return;
    }

    //----------------------------------------------------------
    // Allocate extra alignment padding
    //----------------------------------------------------------

    storage_.resize(bytes + ALIGNMENT);

    //----------------------------------------------------------
    // Reset allocator
    //----------------------------------------------------------

    offset_ = 0;
}

} // namespace renorm