#pragma once

#include "renorm/runtime/compiled_graph.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// CompiledRegionGraphBuilder
//
// Builds the region dependency graph.
//
// Regions
//      ↓
// RegionGraph
//==============================================================

class CompiledRegionGraphBuilder
{
public:

    void build(
        CompiledGraph& compiled) const;

private:

    void connect_regions(
        CompiledGraph& compiled) const;

    void remove_duplicates(
        CompiledGraph& compiled) const;
};

}
}