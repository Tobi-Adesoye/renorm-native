#include "renorm/runtime/simd_region_executor.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void SIMDRegionExecutor::execute(
    VectorFrontier& frontier,
    const CompiledGraph& graph,
    const std::vector<std::size_t>& activeRegions) const
{
    //----------------------------------------------------------
    // Execute every active compiled region
    //----------------------------------------------------------

    for (std::size_t regionId : activeRegions)
    {
        if (regionId >= graph.compiledBlocks.size())
            continue;

        const CompiledBlock& block =
            graph.compiledBlocks[regionId];

        blockExecutor_.execute(
            frontier,
            block);
    }

    //----------------------------------------------------------
    // Advance frontier
    //----------------------------------------------------------

    frontier.swap_buffers();
}

} // namespace runtime
} // namespace renorm