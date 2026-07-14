#include "renorm/runtime/compiled_path_cache.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void CompiledPathCache::clear()
{
    cache.clear();
}

//==============================================================

bool CompiledPathCache::contains(
    NodeId start,
    NodeId goal) const
{
    return
        cache.find(
            make_key(start, goal))
        != cache.end();
}

//==============================================================

void CompiledPathCache::store(
    NodeId start,
    NodeId goal,
    const std::vector<NodeId>& path,
    float cost,
    std::size_t iteration)
{
    CachedPath& entry =
        cache[
            make_key(start, goal)];

    entry.valid = true;

    entry.start = start;

    entry.goal = goal;

    entry.nodes = path;

    entry.cost = cost;

    ++entry.visitCount;

    entry.lastIteration =
        iteration;
}

//==============================================================

CachedPath*
CompiledPathCache::find(
    NodeId start,
    NodeId goal)
{
    auto it =
        cache.find(
            make_key(start, goal));

    if (it == cache.end())
        return nullptr;

    return &it->second;
}

//==============================================================

const CachedPath*
CompiledPathCache::find(
    NodeId start,
    NodeId goal) const
{
    auto it =
        cache.find(
            make_key(start, goal));

    if (it == cache.end())
        return nullptr;

    return &it->second;
}

//==============================================================

CompiledPathCache::Key
CompiledPathCache::make_key(
    NodeId start,
    NodeId goal) const
{
    return
        (static_cast<Key>(start) << 32) |
        static_cast<Key>(goal);
}

} // namespace runtime
} // namespace renorm