#include "pass_region_detection.h"

#include <queue>

namespace renorm
{
namespace compiler
{

std::vector<Region>
PassRegionDetection::run(
    const Graph& graph) const
{
    std::vector<Region> regions;

    const std::size_t N = graph.node_count();

    std::vector<bool> visited(N, false);

    std::size_t nextRegion = 0;

    for (NodeId start = 0; start < N; ++start)
    {
        if (visited[start])
            continue;

        Region region;

        region.id = nextRegion++;

        std::queue<NodeId> q;

        q.push(start);

        visited[start] = true;

        while (!q.empty())
        {
            NodeId node = q.front();

            q.pop();

            region.nodes.push_back(node);

            for (const auto& edge : graph.neighbors(node))
            {
                if (!visited[edge.to])
                {
                    visited[edge.to] = true;

                    q.push(edge.to);
                }
            }
        }

        regions.push_back(region);
    }

    return regions;
}

}
}