#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "renorm/graph.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// VectorFrontier
//
// SIMD-ready frontier representation.
//
// Structure-of-Arrays layout
// minimizes cache misses and allows
// contiguous vector loads.
//
//==============================================================

class VectorFrontier
{
public:

    VectorFrontier() = default;

    explicit VectorFrontier(std::size_t nodeCount);

    //----------------------------------------------------------
    // Memory
    //----------------------------------------------------------

    void resize(std::size_t nodeCount);

    void clear();

    //----------------------------------------------------------
    // Activation
    //----------------------------------------------------------

    void activate(NodeId node);

    bool active(NodeId node) const;

    //----------------------------------------------------------
    // Buffers
    //----------------------------------------------------------

    void swap_buffers();

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    std::size_t active_count() const;

    std::size_t capacity() const;

public:

    //----------------------------------------------------------
    // SIMD contiguous buffers
    //----------------------------------------------------------

    std::vector<float> current;

    std::vector<float> next;

    //----------------------------------------------------------
    // Visited bitmap
    //----------------------------------------------------------

    std::vector<uint8_t> visited;

    //----------------------------------------------------------
    // Parent tree
    //----------------------------------------------------------

    std::vector<NodeId> parent;

    //----------------------------------------------------------
    // Distance
    //----------------------------------------------------------

    std::vector<float> distance;

    //----------------------------------------------------------
    // Depth
    //----------------------------------------------------------

    std::vector<std::uint32_t> depth;

private:

    std::size_t activeNodes = 0;
};

} // namespace runtime
} // namespace renorm