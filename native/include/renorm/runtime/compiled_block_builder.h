#pragma once

#include "renorm/graph.h"
#include "renorm/runtime/compiled_block.h"

#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// CompiledBlockBuilder
//
// Converts a Graph directly into executable CompiledBlocks.
//--------------------------------------------------------------

class CompiledBlockBuilder
{
public:

    //----------------------------------------------------------
    // Build compiled execution blocks.
    //
    // nodeToCompiledBlock[node] -> owning compiled block
    //----------------------------------------------------------

    static std::vector<CompiledBlock> build(
        const Graph& graph,
        std::vector<std::size_t>& nodeToCompiledBlock);

private:

    static constexpr std::size_t DEFAULT_BLOCK_SIZE = 64;
};

} // namespace runtime
} // namespace renorm