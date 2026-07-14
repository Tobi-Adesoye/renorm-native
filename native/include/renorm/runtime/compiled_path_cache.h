#pragma once

#include "renorm/graph.h"

#include <unordered_map>
#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// CachedPath
//
// Stores a previously computed execution path.
//
//==============================================================

struct CachedPath
{
    bool valid = false;

    NodeId start = INVALID_NODE;

    NodeId goal = INVALID_NODE;

    std::vector<NodeId> nodes;

    float cost = 0.0f;

    std::size_t visitCount = 0;

    std::size_t lastIteration = 0;
};

//==============================================================
//
// CompiledPathCache
//
// Runtime cache of previously computed paths.
//
// Used by:
//
//   PredictionEngine
//   RuntimeOptimizer
//   RenormRuntime
//
//==============================================================

class CompiledPathCache
{
public:

    CompiledPathCache() = default;

    //----------------------------------------------------------

    void clear();

    //----------------------------------------------------------

    bool contains(
        NodeId start,
        NodeId goal) const;

    //----------------------------------------------------------

    void store(
        NodeId start,
        NodeId goal,
        const std::vector<NodeId>& path,
        float cost,
        std::size_t iteration);

    //----------------------------------------------------------

    CachedPath* find(
        NodeId start,
        NodeId goal);

    //----------------------------------------------------------

    const CachedPath* find(
        NodeId start,
        NodeId goal) const;

private:

    //----------------------------------------------------------

    using Key = std::uint64_t;

    Key make_key(
        NodeId start,
        NodeId goal) const;

    //----------------------------------------------------------

    std::unordered_map<Key, CachedPath> cache;
};

}
}