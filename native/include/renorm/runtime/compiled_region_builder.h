#pragma once

#include "renorm/runtime/compiled_graph.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// CompiledRegionBuilder
//
// Builds execution regions from compiled blocks.
//
// Phase E
//
// Graph
//    ↓
// Blocks
//    ↓
// Regions
//==============================================================

class CompiledRegionBuilder
{
public:

    void build(
        CompiledGraph& compiled) const;

private:

    static constexpr std::size_t BLOCKS_PER_REGION = 16;

    void assign_regions(
        CompiledGraph& compiled) const;

    void compute_statistics(
        CompiledGraph& compiled) const;
};

}
}