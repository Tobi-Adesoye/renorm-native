#include "renorm/runtime/graph_compiler.h"

#include "renorm/runtime/graph_matrix_builder.h"
#include "renorm/runtime/compiled_region_builder.h"
#include "renorm/runtime/compiled_region_graph_builder.h"

namespace renorm
{
namespace runtime
{

CompiledGraph GraphCompiler::compile(
    const Graph& graph)
{
    //----------------------------------------------------------
    // Output
    //----------------------------------------------------------

    CompiledGraph compiled;

    compiled.graph = &graph;

    //----------------------------------------------------------
    // Phase 1
    // Graph -> Compiled Blocks
    //----------------------------------------------------------

    GraphMatrixBuilder matrixBuilder;

    matrixBuilder.build(
        graph,
        compiled);

    //----------------------------------------------------------
    // Phase 2
    // Blocks -> Regions
    //----------------------------------------------------------

    CompiledRegionBuilder regionBuilder;

    regionBuilder.build(
        compiled);

    //----------------------------------------------------------
    // Phase 3
    // Region connectivity graph
    //----------------------------------------------------------

    CompiledRegionGraphBuilder regionGraphBuilder;

    regionGraphBuilder.build(
        compiled);

    //----------------------------------------------------------
    // Metadata
    //----------------------------------------------------------

    compiled.nodeCount =
        graph.node_count();

    compiled.edgeCount =
        graph.edge_count();

    compiled.blockCount =
        compiled.compiledBlocks.size();

    compiled.regionCount =
        compiled.compiledRegions.size();

    compiled.instructionCount = 0;

    for (const auto& block : compiled.compiledBlocks)
    {
        compiled.instructionCount +=
            block.program.size();
    }

    //----------------------------------------------------------
    // Normalize IDs
    //----------------------------------------------------------

    for (std::size_t i = 0;
         i < compiled.compiledBlocks.size();
         ++i)
    {
        compiled.compiledBlocks[i].id = i;
    }

    for (std::size_t i = 0;
         i < compiled.compiledRegions.size();
         ++i)
    {
        compiled.compiledRegions[i].id = i;
    }

    return compiled;
}

} // namespace runtime
} // namespace renorm