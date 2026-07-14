#include "renorm/runtime/region_cache.h"

namespace renorm
{
namespace runtime
{

bool RegionCache::contains(
    std::size_t region) const
{
    return cache.find(region) != cache.end();
}

const CachedRegion*
RegionCache::lookup(
    std::size_t region) const
{
    auto it = cache.find(region);

    if (it == cache.end())
        return nullptr;

    return &it->second;
}

void RegionCache::store(
    std::size_t region,
    const CachedRegion& data)
{
    cache[region] = data;
}

void RegionCache::clear()
{
    cache.clear();
}

}
}