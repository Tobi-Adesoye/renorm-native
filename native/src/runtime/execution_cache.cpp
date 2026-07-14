#include "renorm/runtime/execution_cache.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void ExecutionCache::initialize(
    const ExecutionContext& context)
{
    if (!context.graph)
        return;

    regionCache.clear();
    blockCache.clear();

    regionCache.resize(
        context.graph->compiledRegions.size());

    blockCache.resize(
        context.graph->compiledBlocks.size());
}

//==============================================================

void ExecutionCache::update(
    ExecutionContext& context)
{
    if (!context.graph)
        return;

    //----------------------------------------------------------
    // Resize if graph changed
    //----------------------------------------------------------

    if (regionCache.size() !=
        context.graph->compiledRegions.size())
    {
        regionCache.resize(
            context.graph->compiledRegions.size());
    }

    if (blockCache.size() !=
        context.graph->compiledBlocks.size())
    {
        blockCache.resize(
            context.graph->compiledBlocks.size());
    }

    //----------------------------------------------------------
    // Cache Regions
    //----------------------------------------------------------

    for (std::size_t i = 0;
         i < context.graph->compiledRegions.size();
         ++i)
    {
        auto& cache =
            regionCache[i];

        const auto& region =
            context.graph->compiledRegions[i];

        cache.valid = true;

        cache.heuristic =
            region.heuristic;

        cache.confidence =
            region.confidence;

        cache.prediction =
            context.frontier.predictionConfidence;

        ++cache.executionCount;

        cache.lastIteration =
            context.iteration;
    }

    //----------------------------------------------------------
    // Cache Blocks
    //----------------------------------------------------------

    for (std::size_t i = 0;
         i < context.graph->compiledBlocks.size();
         ++i)
    {
        auto& cache =
            blockCache[i];

        const auto& block =
            context.graph->compiledBlocks[i];

        cache.valid = true;

        cache.density =
            block.density;

        cache.instructionCount =
            block.instructionCount;

        ++cache.executionCount;

        cache.lastIteration =
            context.iteration;
    }
}

//==============================================================

void ExecutionCache::invalidate()
{
    regionCache.clear();
    blockCache.clear();
}

//==============================================================

bool ExecutionCache::has_region(
    std::size_t regionId) const
{
    return
        regionId < regionCache.size() &&
        regionCache[regionId].valid;
}

//==============================================================

bool ExecutionCache::has_block(
    std::size_t blockId) const
{
    return
        blockId < blockCache.size() &&
        blockCache[blockId].valid;
}

//==============================================================

const CachedRegion&
ExecutionCache::region(
    std::size_t regionId) const
{
    return regionCache[regionId];
}

//==============================================================

const CachedBlock&
ExecutionCache::block(
    std::size_t blockId) const
{
    return blockCache[blockId];
}

} // namespace runtime
} // namespace renorm