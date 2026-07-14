#pragma once

#include "renorm/runtime/execution_context.h"

#include <vector>

namespace renorm
{
namespace runtime
{

//==============================================================
// RegionBFS
//
// Breadth-first traversal over the Region Graph.
//
// Phase F2
//==============================================================

class RegionBFS
{
public:

    bool search(
        ExecutionContext& context,
        std::size_t startRegion,
        std::size_t goalRegion);

private:

    void reconstruct_path(
        ExecutionContext& context,
        std::size_t startRegion,
        std::size_t goalRegion,
        const std::vector<std::size_t>& parent);
};

}
}