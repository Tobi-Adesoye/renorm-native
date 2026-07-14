#include "renorm/runtime/compiled_block_builder.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

std::vector<CompiledBlock>
CompiledBlockBuilder::build(
    const Graph& graph,
    std::vector<std::size_t>& nodeToCompiledBlock)
{
    //----------------------------------------------------------
    // Allocate lookup
    //----------------------------------------------------------

    nodeToCompiledBlock.assign(
        graph.node_count(),
        static_cast<std::size_t>(-1));

    //----------------------------------------------------------
    // Number of blocks
    //----------------------------------------------------------

    const std::size_t blockSize =
        DEFAULT_BLOCK_SIZE;

    const std::size_t blockCount =
        (graph.node_count() + blockSize - 1) /
        blockSize;

    std::vector<CompiledBlock> blocks;

    blocks.resize(blockCount);

    //----------------------------------------------------------
    // Initialize blocks
    //----------------------------------------------------------

    for (std::size_t i = 0; i < blockCount; ++i)
    {
        blocks[i].id = i;
    }

    //----------------------------------------------------------
    // Assign nodes
    //----------------------------------------------------------

    for (NodeId node = 0;
         node < graph.node_count();
         ++node)
    {
        const std::size_t blockId =
            node / blockSize;

        nodeToCompiledBlock[node] =
            blockId;

        blocks[blockId].nodes.push_back(node);
    }

    //----------------------------------------------------------
    // Compile graph edges into instructions
    //----------------------------------------------------------

    for (NodeId from = 0;
         from < graph.node_count();
         ++from)
    {
        const std::size_t owner =
            nodeToCompiledBlock[from];

        const auto& neighbours =
            graph.neighbors(from);

        for (const auto& edge : neighbours)
        {
            //--------------------------------------------------
            // Instruction
            //--------------------------------------------------

            blocks[owner].add_instruction(
                from,
                edge.to,
                edge.weight);

            //--------------------------------------------------
            // Cross-block dependency
            //--------------------------------------------------

            const std::size_t targetBlock =
                nodeToCompiledBlock[
                    edge.to];

            if (targetBlock != owner)
            {
                blocks[owner]
                    .outgoingBlocks
                    .push_back(targetBlock);

                blocks[targetBlock]
                    .incomingBlocks
                    .push_back(owner);
            }
        }
    }

    //----------------------------------------------------------
    // Remove duplicate block links
    //----------------------------------------------------------

    for (CompiledBlock& block : blocks)
    {
        std::sort(
            block.outgoingBlocks.begin(),
            block.outgoingBlocks.end());

        block.outgoingBlocks.erase(
            std::unique(
                block.outgoingBlocks.begin(),
                block.outgoingBlocks.end()),
            block.outgoingBlocks.end());

        std::sort(
            block.incomingBlocks.begin(),
            block.incomingBlocks.end());

        block.incomingBlocks.erase(
            std::unique(
                block.incomingBlocks.begin(),
                block.incomingBlocks.end()),
            block.incomingBlocks.end());

        block.instructionCount =
            block.program.size();

        block.edgeCount =
            block.program.size();

        if (!block.nodes.empty())
        {
            block.density =
                static_cast<float>(
                    block.program.size()) /
                static_cast<float>(
                    block.nodes.size());
        }
    }

    return blocks;
}

} // namespace runtime
} // namespace renorm