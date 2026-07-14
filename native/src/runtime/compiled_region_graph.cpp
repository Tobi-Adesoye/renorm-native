#include "renorm/runtime/compiled_region_graph_builder.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void CompiledRegionGraphBuilder::build(
    CompiledGraph& compiled) const
{
    compiled.regionGraph.clear();

    connect_regions(compiled);

    remove_duplicates(compiled);
}

//==============================================================

void CompiledRegionGraphBuilder::connect_regions(
    CompiledGraph& compiled) const
{
    for (std::size_t blockId = 0;
         blockId < compiled.compiledBlocks.size();
         ++blockId)
    {
        const auto& block =
            compiled.compiledBlocks[blockId];

        const std::size_t sourceRegion =
            compiled.blockToRegion[blockId];

        for (std::size_t neighbourBlock :
             block.outgoingBlocks)
        {
            const std::size_t targetRegion =
                compiled.blockToRegion[
                    neighbourBlock];

            if (sourceRegion == targetRegion)
                continue;

            compiled.compiledRegions[sourceRegion]
                .neighbouringRegions
                .push_back(targetRegion);

            compiled.regionGraph.add_edge(
                sourceRegion,
                targetRegion,
                1.0f);
        }
    }
}

//==============================================================

void CompiledRegionGraphBuilder::remove_duplicates(
    CompiledGraph& compiled) const
{
    for (CompiledRegion& region :
         compiled.compiledRegions)
    {
        auto& neighbours =
            region.neighbouringRegions;

        std::sort(
            neighbours.begin(),
            neighbours.end());

        neighbours.erase(
            std::unique(
                neighbours.begin(),
                neighbours.end()),
            neighbours.end());
    }
}

}
}