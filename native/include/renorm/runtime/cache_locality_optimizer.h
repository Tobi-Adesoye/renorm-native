#pragma once

#include <cstddef>
#include <vector>

namespace renorm
{
namespace runtime
{

class CompiledGraph;
class CompiledBlock;

//==============================================================
//
// CacheLocalityOptimizer
//
// Reorders compiled regions and blocks to maximize
// cache locality during execution.
//
// Goals
//
// • Minimize cache misses
// • Improve spatial locality
// • Improve temporal locality
// • Reduce memory bandwidth
// • Improve SIMD throughput
//
//==============================================================

class CacheLocalityOptimizer
{
public:

    CacheLocalityOptimizer() = default;

    //----------------------------------------------------------
    // Optimize complete graph
    //----------------------------------------------------------

    void optimize(
        CompiledGraph& graph) const;

private:

    //----------------------------------------------------------
    // Block ordering
    //----------------------------------------------------------

    void reorder_blocks(
        CompiledGraph& graph) const;

    //----------------------------------------------------------
    // Instruction ordering
    //----------------------------------------------------------

    void reorder_instructions(
        CompiledBlock& block) const;

    //----------------------------------------------------------
    // Region ordering
    //----------------------------------------------------------

    void reorder_regions(
        CompiledGraph& graph) const;

    //----------------------------------------------------------
    // Build locality score
    //----------------------------------------------------------

    float locality_score(
        const CompiledBlock& block) const;
};

}
}