#include "renorm/search/dijkstra.h"

#include <queue>
#include <vector>
#include <limits>
#include <chrono>
#include <algorithm>

namespace renorm
{

struct DijkstraNode
{
    float distance;
    NodeId node;

    bool operator>(const DijkstraNode& other) const
    {
        return distance > other.distance;
    }
};

SearchResult Dijkstra::run(
    const Graph& graph) const
{
    SearchResult result;

    //----------------------------------------------------------
    // Empty graph
    //----------------------------------------------------------

    if (graph.node_count() == 0)
    {
        return result;
    }

    //----------------------------------------------------------
    // Convention
    //----------------------------------------------------------

    const NodeId start = 0;

    const NodeId goal =
        graph.node_count() - 1;

    //----------------------------------------------------------

    auto begin =
        std::chrono::high_resolution_clock::now();

    //----------------------------------------------------------

    const float INF =
        std::numeric_limits<float>::infinity();

    std::vector<float> distance(
        graph.node_count(),
        INF);

    std::vector<bool> visited(
        graph.node_count(),
        false);

    std::vector<NodeId> parent(
        graph.node_count(),
        static_cast<NodeId>(-1));

    std::priority_queue<
        DijkstraNode,
        std::vector<DijkstraNode>,
        std::greater<DijkstraNode>> frontier;

    //----------------------------------------------------------

    distance[start] = 0.0f;

    frontier.push(
    {
        0.0f,
        start
    });

    //----------------------------------------------------------

    while (!frontier.empty())
    {
        DijkstraNode current =
            frontier.top();

        frontier.pop();

        if (visited[current.node])
        {
            continue;
        }

        visited[current.node] = true;

        result.nodesVisited++;

        //------------------------------------------------------

        if (current.node == goal)
        {
            result.found = true;
            break;
        }

        //------------------------------------------------------

        for (const Edge& edge :
            graph.neighbors(current.node))
        {
            float candidate =
                distance[current.node] +
                edge.weight;

            if (candidate < distance[edge.to])
            {
                distance[edge.to] =
                    candidate;

                parent[edge.to] =
                    current.node;

                frontier.push(
                {
                    candidate,
                    edge.to
                });
            }
        }
    }

    //----------------------------------------------------------
    // Path reconstruction
    //----------------------------------------------------------

    if (result.found)
    {
        NodeId current = goal;

        while (true)
        {
            result.path.push_back(current);

            if (current == start)
            {
                break;
            }

            current = parent[current];
        }

        std::reverse(
            result.path.begin(),
            result.path.end());

        if (!result.path.empty())
        {
            result.pathLength =
                result.path.size() - 1;
        }
    }

    //----------------------------------------------------------

    auto end =
        std::chrono::high_resolution_clock::now();

    result.executionTimeMs =
        std::chrono::duration<double,std::milli>(
            end - begin).count();

    return result;
}

}