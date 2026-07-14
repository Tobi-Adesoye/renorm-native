#include "renorm/runtime/region_bfs.h"

#include <queue>

namespace renorm
{
namespace runtime
{

bool RegionBFS::search(
    ExecutionContext& context,
    std::size_t startRegion,
    std::size_t goalRegion)
{
    const auto regionCount =
        context.compiled.regions.size();

    std::vector<bool> visited(
        regionCount,
        false);

    std::vector<std::size_t> parent(
        regionCount,
        static_cast<std::size_t>(-1));

    std::queue<std::size_t> frontier;

    frontier.push(startRegion);

    visited[startRegion] = true;

    while (!frontier.empty())
    {
        const auto current =
            frontier.front();

        frontier.pop();

        if (current == goalRegion)
        {
            reconstruct_path(
                context,
                startRegion,
                goalRegion,
                parent);

            return true;
        }

        for (const auto next :
             context.compiled
                    .regions[current]
                    .neighbouringRegions)
        {
            if (visited[next])
                continue;

            visited[next] = true;

            parent[next] = current;

            frontier.push(next);
        }
    }

    return false;
}

//==============================================================

void RegionBFS::reconstruct_path(
    ExecutionContext& context,
    std::size_t startRegion,
    std::size_t goalRegion,
    const std::vector<std::size_t>& parent)
{
    context.regionPath.clear();

    std::size_t current =
        goalRegion;

    while (current !=
           static_cast<std::size_t>(-1))
    {
        context.regionPath.push_back(current);

        if (current == startRegion)
            break;

        current = parent[current];
    }

    std::reverse(
        context.regionPath.begin(),
        context.regionPath.end());
}

}
}