#include "renorm/runtime/graph_matrix_builder.h"

#include "renorm/runtime/blocked_matrix_builder.h"
#include "renorm/runtime/compiled_block_builder.h"

namespace renorm
{
namespace runtime
{

void GraphMatrixBuilder::build(
    const Graph& graph,
    CompiledGraph& compiled) const
{
    //----------------------------------------------------------
    // Reset previous compilation
    //----------------------------------------------------------

    compiled.clear();

    //----------------------------------------------------------
    // Preserve original graph
    //----------------------------------------------------------

    compiled.graph = &graph;

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    compiled.nodeCount = graph.node_count();

    //----------------------------------------------------------
    // Phase C1
    //
    // Build executable blocks
    //----------------------------------------------------------

    compiled.compiledBlocks =
        CompiledBlockBuilder::build(
            graph,
            compiled.nodeToCompiledBlock);

    compiled.blockCount =
        compiled.compiledBlocks.size();

    //----------------------------------------------------------
    // Count instructions
    //----------------------------------------------------------

    compiled.instructionCount = 0;

    for (const CompiledBlock& block :
         compiled.compiledBlocks)
    {
        compiled.instructionCount +=
            block.program.size();
    }

    //----------------------------------------------------------
    // Compatibility layer
    //
    // Keep BlockedMatrix alive until Phase C2.
    //----------------------------------------------------------

    compiled.blocked =
        BlockedMatrixBuilder::build(graph);

    compiled.edgeCount =
        compiled.blocked.totalNonZeros;
}

} // namespace runtime
} // namespace renorm