#include "renorm/runtime/region_astar.h"

#include <queue>
#include <limits>
#include <algorithm>

namespace renorm
{
namespace runtime
{

struct RegionQueueNode
{
    std::size_t region;
    float f;

    bool operator<(const RegionQueueNode& rhs) const
    {
        return f > rhs.f;
    }
};

//==============================================================

bool RegionAStar::search(
    ExecutionContext& context,
    std::size_t startRegion,
    std::size_t goalRegion)
{
    const std::size_t n =
        context.compiled.regions.size();

    std::priority_queue<RegionQueueNode> open;

    std::vector<float> g(
        n,
        std::numeric_limits<float>::infinity());

    std::vector<float> f(
        n,
        std::numeric_limits<float>::infinity());

    std::vector<std::size_t> parent(
        n,
        static_cast<std::size_t>(-1));

    std::vector<bool> closed(
        n,
        false);

    g[startRegion] = 0.0f;

    f[startRegion] =
        heuristic(context, startRegion);

    open.push(
    {
        startRegion,
        f[startRegion]
    });

    while (!open.empty())
    {
        auto current =
            open.top().region;

        open.pop();

        if (closed[current])
            continue;

        closed[current] = true;

        ++context.statistics.regionsVisited;

        if (current == goalRegion)
        {
            reconstruct_path(
                context,
                startRegion,
                goalRegion,
                parent);

            context.statistics.regionPathLength =
                context.regionPath.size();

            return true;
        }

        for (auto next :
             context.compiled
                    .regions[current]
                    .neighbouringRegions)
        {
            if (closed[next])
                continue;

            float tentative =
                g[current] + 1.0f;

            if (tentative < g[next])
            {
                parent[next] = current;

                g[next] = tentative;

                f[next] =
                    tentative +
                    heuristic(context, next);

                open.push(
                {
                    next,
                    f[next]
                });
            }
        }
    }

    return false;
}

//==============================================================

float RegionAStar::heuristic(
    const ExecutionContext& context,
    std::size_t region) const
{
    const auto& r =
        context.compiled.regions[region];

    //----------------------------------------------------------
    // Composite heuristic
    //----------------------------------------------------------

    float h = 0.0f;

    h += r.heuristic;

    h -= r.confidence;

    h +=
        static_cast<float>(
            r.neighbouringRegions.size()) * 0.1f;

    return h;
}

//==============================================================

void RegionAStar::reconstruct_path(
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