#pragma once

#include "renorm/runtime/compiled_graph.h"

#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// RegionFrontier
//
// Active execution frontier operating on regions.
//
// Region
//   ↓
// Block
//   ↓
// Instruction
//--------------------------------------------------------------

class RegionFrontier
{
public:

    //----------------------------------------------------------
    // Build initial region frontier
    //----------------------------------------------------------

    void build(
        const CompiledGraph& compiled,
        std::size_t startRegion);

    //----------------------------------------------------------
    // Frontier operations
    //----------------------------------------------------------

    bool empty() const;

    std::size_t next();

    void activate(
        std::size_t region);

    bool active(
        std::size_t region) const;

    void clear();

private:

    std::vector<bool> activeRegions;

    std::vector<std::size_t> queue;

    std::size_t head = 0;
};

}
}