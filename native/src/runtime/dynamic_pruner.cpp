#include "renorm/runtime/dynamic_pruner.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void DynamicPruner::prune(
    ExecutionContext& context)
{
    if (!context.graph)
        return;

    prune_regions(
        context);

    prune_blocks(
        context);
}

//==============================================================

void DynamicPruner::prune_regions(
    ExecutionContext& context)
{
    std::size_t removed = 0;

    for (auto& region :
         context.graph->compiledRegions)
    {
        region.active =
            keep_region(
                context,
                region);

        if (!region.active)
            ++removed;
    }

    if (!context.graph->compiledRegions.empty())
    {
        context.pruningRatio =
            static_cast<float>(removed) /
            static_cast<float>(
                context.graph->compiledRegions.size());
    }

    context.result.expansionReduction =
        context.pruningRatio;
}

//==============================================================

void DynamicPruner::prune_blocks(
    ExecutionContext& context)
{
    context.frontier.activeBlocks.clear();

    for (std::size_t blockId = 0;
         blockId <
         context.graph->compiledBlocks.size();
         ++blockId)
    {
        const auto& block =
            context.graph->compiledBlocks[
                blockId];

        if (keep_block(
                context,
                block))
        {
            context.frontier.activeBlocks.push_back(
                blockId);
        }
    }
}

//==============================================================

bool DynamicPruner::keep_region(
    const ExecutionContext& context,
    const CompiledRegion& region) const
{
    float score = 0.0f;

    score += region.confidence;

    score += region.heuristic;

    score += region.density;

    score +=
        context.frontier.predictionConfidence;

    score +=
        context.frontier.heuristicScore;

    score /= 5.0f;

    return score >= 0.35f;
}

//==============================================================

bool DynamicPruner::keep_block(
    const ExecutionContext& context,
    const CompiledBlock& block) const
{
    if (block.program.empty())
        return false;

    float density =
        block.density;

    float prediction =
        context.frontier.predictionConfidence;

    float heuristic =
        context.frontier.heuristicScore;

    float score =
        (density +
         prediction +
         heuristic) / 3.0f;

    return score >= 0.30f;
}

} // namespace runtime
} // namespace renorm