#include "renorm/runtime/region_analyser.h"

namespace renorm
{
namespace runtime
{

void RegionAnalyser::analyse(
    ExecutionContext& context)
{
    compute_density(context);

    compute_connectivity(context);

    compute_branching(context);

    compute_boundary(context);
}

//==============================================================

void RegionAnalyser::compute_density(
    ExecutionContext& context)
{
    for (auto& region : context.compiled.regions)
    {
        std::size_t nodes = 0;
        std::size_t edges = 0;

        for (auto blockId : region.blocks)
        {
            const auto& block =
                context.compiled.compiledBlocks[blockId];

            nodes += block.nodes.size();
            edges += block.program.size();
        }

        if (nodes)
            region.density =
                static_cast<float>(edges) /
                static_cast<float>(nodes);
        else
            region.density = 0.0f;
    }
}

//==============================================================

void RegionAnalyser::compute_connectivity(
    ExecutionContext& context)
{
    (void)context;

    // Phase E2
}

//==============================================================

void RegionAnalyser::compute_branching(
    ExecutionContext& context)
{
    (void)context;

    // Phase E2
}

//==============================================================

void RegionAnalyser::compute_boundary(
    ExecutionContext& context)
{
    (void)context;

    // Phase E2
}

}
}