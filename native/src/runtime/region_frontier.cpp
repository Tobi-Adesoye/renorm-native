#include "renorm/runtime/region_frontier.h"

namespace renorm
{
namespace runtime
{

void RegionFrontier::build(
    const CompiledGraph& compiled,
    std::size_t startRegion)
{
    clear();

    activeRegions.resize(
        compiled.regionCount,
        false);

    if (startRegion < compiled.regionCount)
    {
        activeRegions[startRegion] = true;
        queue.push_back(startRegion);
    }
}

bool RegionFrontier::empty() const
{
    return head >= queue.size();
}

std::size_t RegionFrontier::next()
{
    return queue[head++];
}

void RegionFrontier::activate(
    std::size_t region)
{
    if (region >= activeRegions.size())
        return;

    if (activeRegions[region])
        return;

    activeRegions[region] = true;
    queue.push_back(region);
}

bool RegionFrontier::active(
    std::size_t region) const
{
    if (region >= activeRegions.size())
        return false;

    return activeRegions[region];
}

void RegionFrontier::clear()
{
    activeRegions.clear();
    queue.clear();
    head = 0;
}

}
}