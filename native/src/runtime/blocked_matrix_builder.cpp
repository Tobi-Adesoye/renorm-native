#include "renorm/runtime/blocked_matrix_builder.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

BlockedMatrix BlockedMatrixBuilder::build(
    const Graph& graph)
{
    constexpr std::size_t DEFAULT_BLOCK_SIZE = 64;

    BlockedMatrix blocked;

    blocked.clear();

    blocked.rows = graph.node_count();
    blocked.columns = graph.node_count();
    blocked.blockSize = DEFAULT_BLOCK_SIZE;

    //----------------------------------------------------------
    // Allocate lookup
    //----------------------------------------------------------

    blocked.nodeToBlock.assign(
        graph.node_count(),
        static_cast<std::size_t>(-1));

    //----------------------------------------------------------
    // Create blocks
    //----------------------------------------------------------

    std::size_t nextBlockId = 0;

    for (std::size_t first = 0;
         first < graph.node_count();
         first += DEFAULT_BLOCK_SIZE)
    {
        MatrixBlock& block =
            blocked.add_block();

        block.id = nextBlockId++;

        const std::size_t last =
            std::min(
                first + DEFAULT_BLOCK_SIZE,
                graph.node_count());

        block.rowCount =
            last - first;

        block.columnCount =
            graph.node_count();

        block.rows.reserve(block.rowCount);

        for (NodeId node = first;
             node < last;
             ++node)
        {
            block.rows.push_back(node);

            blocked.nodeToBlock[node] =
                block.id;
        }

        block.columns.resize(
            graph.node_count());

        for (NodeId i = 0;
             i < graph.node_count();
             ++i)
        {
            block.columns[i] = i;
        }

        block.values.assign(
            block.rowCount *
            block.columnCount,
            0.0f);

        //------------------------------------------------------
        // Build dense matrix + compiled edge list
        //------------------------------------------------------

        for (std::size_t r = 0;
             r < block.rowCount;
             ++r)
        {
            NodeId u =
                block.rows[r];

            for (const Edge& edge :
                 graph.neighbors(u))
            {
                block.at(
                    r,
                    edge.to) = edge.weight;

                block.edges.push_back(
                {
                    u,
                    edge.to,
                    edge.weight
                });

                block.nonZeros++;

                blocked.totalNonZeros++;
            }
        }

        //------------------------------------------------------
        // Density
        //------------------------------------------------------

        const std::size_t cells =
            block.rowCount *
            block.columnCount;

        if (cells != 0)
        {
            block.density =
                static_cast<float>(block.nonZeros) /
                static_cast<float>(cells);
        }
    }

    //----------------------------------------------------------
    // Build block graph
    //----------------------------------------------------------

    for (NodeId u = 0;
         u < graph.node_count();
         ++u)
    {
        const std::size_t sourceBlock =
            blocked.block_of(u);

        for (const Edge& edge :
             graph.neighbors(u))
        {
            const std::size_t destBlock =
                blocked.block_of(edge.to);

            if (sourceBlock == destBlock)
                continue;

            auto& outgoing =
                blocked.blocks[sourceBlock]
                    .outgoingBlocks;

            if (std::find(
                    outgoing.begin(),
                    outgoing.end(),
                    destBlock) == outgoing.end())
            {
                outgoing.push_back(destBlock);
            }

            auto& incoming =
                blocked.blocks[destBlock]
                    .incomingBlocks;

            if (std::find(
                    incoming.begin(),
                    incoming.end(),
                    sourceBlock) == incoming.end())
            {
                incoming.push_back(sourceBlock);
            }
        }
    }

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    blocked.blockCount =
        blocked.blocks.size();

    if (!blocked.blocks.empty())
    {
        float density = 0.0f;

        for (const auto& block :
             blocked.blocks)
        {
            density +=
                block.density;
        }

        blocked.averageDensity =
            density /
            static_cast<float>(
                blocked.blocks.size());
    }

    return blocked;
}

} // namespace runtime
} // namespace renorm