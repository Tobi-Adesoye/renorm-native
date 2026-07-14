#pragma once

#include "renorm/runtime/execution_context.h"

#include <vector>
#include <unordered_set>

namespace renorm
{
namespace runtime
{

//==============================================================
// RegionGraph
//
// Builds connectivity between compiled regions.
//
// Phase F1
//==============================================================

class RegionGraph
{
public:

    void build(
        ExecutionContext& context);

private:

    void build_region_edges(
        ExecutionContext& context);

    void remove_duplicate_edges(
        ExecutionContext& context);
};

}
}