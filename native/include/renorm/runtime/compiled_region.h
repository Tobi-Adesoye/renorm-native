#pragma once

#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// CompiledRegion
//
// Groups multiple CompiledBlocks into a higher-level execution
// unit.
//
// Runtime hierarchy:
//
// Graph
//    ↓
// Regions
//    ↓
// Blocks
//    ↓
// Instructions
//
// Regions are the unit used by the reasoning engine for
// coarse-grained scheduling and pruning.
//--------------------------------------------------------------

struct CompiledRegion
{
    //----------------------------------------------------------
    // Runtime identifier
    //----------------------------------------------------------

    std::size_t id = 0;

    //----------------------------------------------------------
    // Blocks owned by this region
    //----------------------------------------------------------

    std::vector<std::size_t> blocks;

    //----------------------------------------------------------
    // Region dependency graph
    //----------------------------------------------------------

    std::vector<std::size_t> neighbouringRegions;

    //----------------------------------------------------------
    // Region statistics
    //----------------------------------------------------------

    float density = 0.0f;

    float heuristic = 0.0f;

    float confidence = 0.0f;

    //----------------------------------------------------------
    // Runtime state
    //----------------------------------------------------------

    bool active = true;

    //----------------------------------------------------------
    // Helpers
    //----------------------------------------------------------

    inline void reserve_blocks(std::size_t count)
    {
        blocks.reserve(count);
    }

    inline void reserve_neighbours(std::size_t count)
    {
        neighbouringRegions.reserve(count);
    }

    inline void add_block(std::size_t blockId)
    {
        blocks.push_back(blockId);
    }

    inline void add_neighbour(std::size_t regionId)
    {
        neighbouringRegions.push_back(regionId);
    }

    inline std::size_t block_count() const
    {
        return blocks.size();
    }

    inline std::size_t neighbour_count() const
    {
        return neighbouringRegions.size();
    }
};

} // namespace runtime
} // namespace renorm