#pragma once

#include "renorm/runtime/execution_context.h"

#include <vector>

namespace renorm
{
namespace runtime
{

//==============================================================
// RegionAStar
//
// Heuristic search over the Region Graph.
//
// Phase F3
//==============================================================

class RegionAStar
{
public:

    bool search(
        ExecutionContext& context,
        std::size_t startRegion,
        std::size_t goalRegion);

private:

    float heuristic(
        const ExecutionContext& context,
        std::size_t region) const;

    void reconstruct_path(
        ExecutionContext& context,
        std::size_t startRegion,
        std::size_t goalRegion,
        const std::vector<std::size_t>& parent);
};

}
}