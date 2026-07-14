#include "renorm/runtime/compiled_region_builder.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void CompiledRegionBuilder::build(
    CompiledGraph& compiled) const
{
    compiled.compiledRegions.clear();

    compiled.blockToRegion.assign(
        compiled.compiledBlocks.size(),
        static_cast<std::size_t>(-1));

    assign_regions(compiled);

    compute_statistics(compiled);

    compiled.regionCount =
        compiled.compiledRegions.size();
}

//==============================================================

void CompiledRegionBuilder::assign_regions(
    CompiledGraph& compiled) const
{
    if (compiled.compiledBlocks.empty())
        return;

    const std::size_t regionCount =
        (compiled.compiledBlocks.size() +
         BLOCKS_PER_REGION - 1) /
        BLOCKS_PER_REGION;

    compiled.compiledRegions.resize(regionCount);

    for (std::size_t r = 0;
         r < regionCount;
         ++r)
    {
        compiled.compiledRegions[r].id = r;
    }

    for (std::size_t block = 0;
         block < compiled.compiledBlocks.size();
         ++block)
    {
        const std::size_t region =
            block / BLOCKS_PER_REGION;

        compiled.blockToRegion[block] =
            region;

        compiled.compiledRegions[region]
            .blocks
            .push_back(block);
    }
}

//==============================================================

void CompiledRegionBuilder::compute_statistics(
    CompiledGraph& compiled) const
{
    for (CompiledRegion& region :
         compiled.compiledRegions)
    {
        float density = 0.0f;

        float heuristic = 0.0f;

        std::size_t edges = 0;

        for (std::size_t blockId :
             region.blocks)
        {
            const auto& block =
                compiled.compiledBlocks[blockId];

            density += block.density;

            heuristic +=
                static_cast<float>(
                    block.edgeCount);

            edges += block.edgeCount;
        }

        if (!region.blocks.empty())
        {
            density /=
                static_cast<float>(
                    region.blocks.size());

            heuristic /=
                static_cast<float>(
                    region.blocks.size());
        }

        region.density = density;

        region.heuristic = heuristic;

        region.confidence =
            density > 0.0f
                ? density
                : 1.0f;

        region.active = true;

        (void)edges;
    }
}

}
}