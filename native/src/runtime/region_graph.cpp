#include "renorm/runtime/region_graph.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

void RegionGraph::build(
    ExecutionContext& context)
{
    build_region_edges(context);

    remove_duplicate_edges(context);
}

//==============================================================

void RegionGraph::build_region_edges(
    ExecutionContext& context)
{
    auto& graph =
        context.compiled;

    for (auto& region :
         graph.regions)
    {
        region.neighbouringRegions.clear();
    }

    //----------------------------------------------------------
    // Walk every compiled block
    //----------------------------------------------------------

    for (const auto& block :
         graph.compiledBlocks)
    {
        const auto ownerRegion =
            graph.blockToRegion[block.id];

        for (const auto targetBlock :
             block.outgoingBlocks)
        {
            const auto targetRegion =
                graph.blockToRegion[targetBlock];

            if (ownerRegion == targetRegion)
                continue;

            graph.regions[ownerRegion]
                .neighbouringRegions
                .push_back(targetRegion);
        }
    }
}

//==============================================================

void RegionGraph::remove_duplicate_edges(
    ExecutionContext& context)
{
    for (auto& region :
         context.compiled.regions)
    {
        auto& edges =
            region.neighbouringRegions;

        std::sort(
            edges.begin(),
            edges.end());

        edges.erase(
            std::unique(
                edges.begin(),
                edges.end()),
            edges.end());
    }
}

}
}