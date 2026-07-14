#include "renorm/runtime/block_scheduler.h"

namespace renorm
{
namespace runtime
{

void BlockScheduler::schedule(
    ExecutionContext& context)
{
    if (!context.graph)
        return;

    const auto& region =
        context.graph->compiledRegions[
            context.currentRegion];

    for (auto block : region.blocks)
    {
        if (!should_execute(
                context,
                block))
            continue;

        context.frontier.activeBlocks.push_back(
            block);
    }
}

bool BlockScheduler::should_execute(
    const ExecutionContext&,
    std::size_t)
const
{
    // Phase C:
    // Execute every block.
    // Phase D:
    // reasoning-based pruning.

    return true;
}

float BlockScheduler::compute_priority(
    const ExecutionContext&,
    std::size_t)
const
{
    return 1.0f;
}

}
}