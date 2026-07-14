#include "renorm/runtime/cache_locality_optimizer.h"

#include "renorm/runtime/compiled_graph.h"
#include "renorm/runtime/compiled_block.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void CacheLocalityOptimizer::optimize(
    CompiledGraph& graph) const
{
    reorder_blocks(graph);

    reorder_regions(graph);

    for (auto& block :
         graph.compiledBlocks)
    {
        reorder_instructions(block);
    }
}

//==============================================================

void CacheLocalityOptimizer::reorder_blocks(
    CompiledGraph& graph) const
{
    std::stable_sort(

        graph.compiledBlocks.begin(),

        graph.compiledBlocks.end(),

        [this](
            const CompiledBlock& a,
            const CompiledBlock& b)
        {
            return
                locality_score(a) >
                locality_score(b);
        });
}

//==============================================================

void CacheLocalityOptimizer::reorder_regions(
    CompiledGraph& graph) const
{
    std::stable_sort(

        graph.compiledRegions.begin(),

        graph.compiledRegions.end(),

        [](const auto& a,
           const auto& b)
        {
            return
                a.blocks.size() >
                b.blocks.size();
        });
}

//==============================================================

void CacheLocalityOptimizer::reorder_instructions(
    CompiledBlock& block) const
{
    std::stable_sort(

        block.program.begin(),

        block.program.end(),

        [](const ExecutionInstruction& lhs,
           const ExecutionInstruction& rhs)
        {
            //--------------------------------------------------
            // Group instructions by source node
            //--------------------------------------------------

            if (lhs.from != rhs.from)
                return lhs.from < rhs.from;

            //--------------------------------------------------
            // Then by destination
            //--------------------------------------------------

            return lhs.to < rhs.to;
        });
}

//==============================================================

float CacheLocalityOptimizer::locality_score(
    const CompiledBlock& block) const
{
    float score = 0.0f;

    //----------------------------------------------------------
    // Dense blocks execute better
    //----------------------------------------------------------

    score += block.density;

    //----------------------------------------------------------
    // Larger instruction streams
    //----------------------------------------------------------

    score +=
        static_cast<float>(
            block.program.size()) *
        0.01f;

    //----------------------------------------------------------
    // Fewer incoming edges
    //----------------------------------------------------------

    score +=
        1.0f /
        (1.0f +
         static_cast<float>(
             block.incomingBlocks.size()));

    //----------------------------------------------------------
    // Fewer outgoing edges
    //----------------------------------------------------------

    score +=
        1.0f /
        (1.0f +
         static_cast<float>(
             block.outgoingBlocks.size()));

    return score;
}

} // namespace runtime
} // namespace renorm