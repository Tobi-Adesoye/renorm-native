#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// RegionFusion
//
// Performs coarse-grained runtime region fusion.
//
// Graph
//      ↓
// Regions
//      ↓
// Larger Regions
//
//==============================================================

class RegionFusion
{
public:

    RegionFusion() = default;

    //----------------------------------------------------------

    void fuse(
        ExecutionContext& context);

private:

    bool can_fuse(
        const CompiledRegion& left,
        const CompiledRegion& right) const;

    void merge(
        ExecutionContext& context,
        std::size_t leftRegion,
        std::size_t rightRegion);
};

}
}