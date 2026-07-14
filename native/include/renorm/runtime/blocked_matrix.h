#pragma once

#include "renorm/graph.h"

#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// Compiled execution edge
//--------------------------------------------------------------

struct BlockEdge
{
    NodeId from = INVALID_NODE;
    NodeId to   = INVALID_NODE;

    float weight = 0.0f;
};

//--------------------------------------------------------------
// MatrixBlock
//--------------------------------------------------------------

struct MatrixBlock
{
    //----------------------------------------------------------
    // Runtime ID
    //----------------------------------------------------------

    std::size_t id = 0;

    //----------------------------------------------------------
    // Node mapping
    //----------------------------------------------------------

    std::vector<NodeId> rows;
    std::vector<NodeId> columns;

    //----------------------------------------------------------
    // Dense matrix
    //----------------------------------------------------------

    std::vector<float> values;

    //----------------------------------------------------------
    // Stage B1
    //
    // Precompiled execution edges
    //----------------------------------------------------------

    std::vector<BlockEdge> edges;

    //----------------------------------------------------------
    // Stage B1
    //
    // Block graph
    //----------------------------------------------------------

    std::vector<std::size_t> outgoingBlocks;
    std::vector<std::size_t> incomingBlocks;

    //----------------------------------------------------------
    // Dimensions
    //----------------------------------------------------------

    std::size_t rowCount = 0;
    std::size_t columnCount = 0;

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    std::size_t nonZeros = 0;

    float density = 0.0f;

    //----------------------------------------------------------
    // Dense access
    //----------------------------------------------------------

    inline float& at(
        std::size_t r,
        std::size_t c)
    {
        return values[r * columnCount + c];
    }

    inline const float& at(
        std::size_t r,
        std::size_t c) const
    {
        return values[r * columnCount + c];
    }
};

//--------------------------------------------------------------
// BlockedMatrix
//--------------------------------------------------------------

class BlockedMatrix
{
public:

    //----------------------------------------------------------
    // Matrix storage
    //----------------------------------------------------------

    std::vector<MatrixBlock> blocks;

    //----------------------------------------------------------
    // Fast lookup
    //----------------------------------------------------------

    std::vector<std::size_t> nodeToBlock;

    //----------------------------------------------------------
    // Dimensions
    //----------------------------------------------------------

    std::size_t rows = 0;
    std::size_t columns = 0;

    std::size_t blockSize = 64;

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    std::size_t blockCount = 0;

    std::size_t totalNonZeros = 0;

    float averageDensity = 0.0f;

    //----------------------------------------------------------
    // Management
    //----------------------------------------------------------

    inline void clear()
    {
        blocks.clear();
        nodeToBlock.clear();

        rows = 0;
        columns = 0;

        blockSize = 64;

        blockCount = 0;
        totalNonZeros = 0;

        averageDensity = 0.0f;
    }

    inline void reserve(
        std::size_t count)
    {
        blocks.reserve(count);
    }

    inline MatrixBlock& add_block()
    {
        blocks.emplace_back();
        return blocks.back();
    }

    inline bool empty() const
    {
        return blocks.empty();
    }

    inline std::size_t size() const
    {
        return blocks.size();
    }

    //----------------------------------------------------------
    // Lookup
    //----------------------------------------------------------

    inline std::size_t block_of(
        NodeId node) const
    {
        return nodeToBlock[node];
    }

    //----------------------------------------------------------
    // Access
    //----------------------------------------------------------

    inline MatrixBlock& operator[](
        std::size_t index)
    {
        return blocks[index];
    }

    inline const MatrixBlock& operator[](
        std::size_t index) const
    {
        return blocks[index];
    }
};

} // namespace runtime
} // namespace renorm