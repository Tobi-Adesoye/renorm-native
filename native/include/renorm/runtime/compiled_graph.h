#pragma once

#include "renorm/graph.h"

#include "renorm/runtime/blocked_matrix.h"
#include "renorm/runtime/compiled_block.h"
#include "renorm/runtime/compiled_region.h"
#include "renorm/runtime/compiled_region_graph.h"

#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

struct CompiledGraph
{
    //----------------------------------------------------------
    // Original graph
    //----------------------------------------------------------

    const Graph* graph = nullptr;

    //----------------------------------------------------------
    // Legacy matrix
    //----------------------------------------------------------

    BlockedMatrix blocked;

    //----------------------------------------------------------
    // Blocks
    //----------------------------------------------------------

    std::vector<CompiledBlock> compiledBlocks;

    //----------------------------------------------------------
    // Regions
    //----------------------------------------------------------

    std::vector<CompiledRegion> compiledRegions;

    //----------------------------------------------------------
    // Region graph
    //----------------------------------------------------------

    CompiledRegionGraph regionGraph;

    //----------------------------------------------------------
    // Lookup tables
    //----------------------------------------------------------

    std::vector<std::size_t> nodeToCompiledBlock;

    std::vector<std::size_t> blockToRegion;

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    std::size_t nodeCount = 0;

    std::size_t edgeCount = 0;

    std::size_t blockCount = 0;

    std::size_t regionCount = 0;

    std::size_t instructionCount = 0;

    //----------------------------------------------------------
    // Helpers
    //----------------------------------------------------------

    inline void clear()
    {
        graph = nullptr;

        blocked.clear();

        compiledBlocks.clear();

        compiledRegions.clear();

        regionGraph.clear();

        nodeToCompiledBlock.clear();

        blockToRegion.clear();

        nodeCount = 0;
        edgeCount = 0;
        blockCount = 0;
        regionCount = 0;
        instructionCount = 0;
    }

    inline bool empty() const
    {
        return compiledBlocks.empty();
    }

    inline std::size_t size() const
    {
        return compiledBlocks.size();
    }

    inline CompiledBlock& operator[](std::size_t i)
    {
        return compiledBlocks[i];
    }

    inline const CompiledBlock& operator[](std::size_t i) const
    {
        return compiledBlocks[i];
    }

    //----------------------------------------------------------
    // Region helpers
    //----------------------------------------------------------

    inline CompiledRegion& region(std::size_t i)
    {
        return compiledRegions[i];
    }

    inline const CompiledRegion& region(std::size_t i) const
    {
        return compiledRegions[i];
    }

    //----------------------------------------------------------
    // Region count
    //----------------------------------------------------------

    inline std::size_t regions() const
    {
        return compiledRegions.size();
    }

    //----------------------------------------------------------
    // Block count
    //----------------------------------------------------------

    inline std::size_t blocks() const
    {
        return compiledBlocks.size();
    }
};

}
}