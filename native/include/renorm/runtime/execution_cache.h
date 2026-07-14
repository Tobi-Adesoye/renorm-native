#pragma once

#include "renorm/runtime/execution_context.h"

#include <unordered_map>
#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// CachedRegion
//
//==============================================================

struct CachedRegion
{
    bool valid = false;

    float heuristic = 0.0f;

    float confidence = 0.0f;

    float prediction = 0.0f;

    std::size_t executionCount = 0;

    std::size_t lastIteration = 0;
};

//==============================================================
//
// CachedBlock
//
//==============================================================

struct CachedBlock
{
    bool valid = false;

    float density = 0.0f;

    std::size_t executionCount = 0;

    std::size_t instructionCount = 0;

    std::size_t lastIteration = 0;
};

//==============================================================
//
// ExecutionCache
//
// Runtime optimization cache.
//
// Stores reusable execution information between iterations.
//
//==============================================================

class ExecutionCache
{
public:

    ExecutionCache() = default;

    //----------------------------------------------------------

    void initialize(
        const ExecutionContext& context);

    //----------------------------------------------------------

    void update(
        ExecutionContext& context);

    //----------------------------------------------------------

    void invalidate();

    //----------------------------------------------------------

    bool has_region(
        std::size_t regionId) const;

    bool has_block(
        std::size_t blockId) const;

    //----------------------------------------------------------

    const CachedRegion&
    region(
        std::size_t regionId) const;

    const CachedBlock&
    block(
        std::size_t blockId) const;

private:

    std::vector<CachedRegion> regionCache;

    std::vector<CachedBlock> blockCache;
};

}
}