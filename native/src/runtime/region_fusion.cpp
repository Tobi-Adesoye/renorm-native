#include "renorm/runtime/region_fusion.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void RegionFusion::fuse(
    ExecutionContext& context)
{
    if (!context.graph)
        return;

    auto& regions =
        context.graph->compiledRegions;

    if (regions.size() < 2)
        return;

    //----------------------------------------------------------

    std::size_t i = 0;

    while (i + 1 < regions.size())
    {
        if (can_fuse(
                regions[i],
                regions[i + 1]))
        {
            merge(
                context,
                i,
                i + 1);

            regions.erase(
                regions.begin() + i + 1);

            continue;
        }

        ++i;
    }

    //----------------------------------------------------------
    // Update region count
    //----------------------------------------------------------

    context.graph->regionCount =
        regions.size();
}

//==============================================================

bool RegionFusion::can_fuse(
    const CompiledRegion& left,
    const CompiledRegion& right) const
{
    //----------------------------------------------------------
    // Both must be active
    //----------------------------------------------------------

    if (!left.active)
        return false;

    if (!right.active)
        return false;

    //----------------------------------------------------------
    // Prevent oversized regions
    //----------------------------------------------------------

    if (left.blocks.size() +
        right.blocks.size() >
        64)
    {
        return false;
    }

    //----------------------------------------------------------
    // Heuristic similarity
    //----------------------------------------------------------

    if (std::abs(
            left.heuristic -
            right.heuristic) > 0.25f)
    {
        return false;
    }

    return true;
}

//==============================================================

void RegionFusion::merge(
    ExecutionContext& context,
    std::size_t leftIndex,
    std::size_t rightIndex)
{
    auto& regions =
        context.graph->compiledRegions;

    auto& left =
        regions[leftIndex];

    auto& right =
        regions[rightIndex];

    //----------------------------------------------------------
    // Merge blocks
    //----------------------------------------------------------

    left.blocks.insert(
        left.blocks.end(),
        right.blocks.begin(),
        right.blocks.end());

    //----------------------------------------------------------
    // Merge neighbours
    //----------------------------------------------------------

    left.neighbouringRegions.insert(
        left.neighbouringRegions.end(),
        right.neighbouringRegions.begin(),
        right.neighbouringRegions.end());

    //----------------------------------------------------------
    // Remove duplicate neighbours
    //----------------------------------------------------------

    std::sort(
        left.neighbouringRegions.begin(),
        left.neighbouringRegions.end());

    left.neighbouringRegions.erase(
        std::unique(
            left.neighbouringRegions.begin(),
            left.neighbouringRegions.end()),
        left.neighbouringRegions.end());

    //----------------------------------------------------------
    // Remove self-reference
    //----------------------------------------------------------

    left.neighbouringRegions.erase(
        std::remove(
            left.neighbouringRegions.begin(),
            left.neighbouringRegions.end(),
            left.id),
        left.neighbouringRegions.end());

    //----------------------------------------------------------
    // Update statistics
    //----------------------------------------------------------

    left.density =
        (left.density +
         right.density) * 0.5f;

    left.heuristic =
        (left.heuristic +
         right.heuristic) * 0.5f;

    left.confidence =
        (left.confidence +
         right.confidence) * 0.5f;

    //----------------------------------------------------------
    // Runtime metrics
    //----------------------------------------------------------

    ++context.executedRegions;

    context.executionEfficiency *=
        1.01f;

    if (context.executionEfficiency > 1.0f)
        context.executionEfficiency = 1.0f;

    context.regionsRescheduled = true;
}

} // namespace runtime
} // namespace renorm