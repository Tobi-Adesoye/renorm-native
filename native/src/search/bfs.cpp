#include "renorm/search/bfs.h"

#include <queue>
#include <chrono>
#include <algorithm>

namespace renorm
{

SearchResult BFS::run(
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
    // Assumption
    //
    // Current graph convention:
    //
    // start = node 0
    // goal  = last node
    //----------------------------------------------------------

    const NodeId start = 0;

    const NodeId goal =
        graph.node_count() - 1;

    //----------------------------------------------------------

    auto begin =
        std::chrono::high_resolution_clock::now();

    //----------------------------------------------------------

    std::vector<bool> visited(
        graph.node_count(),
        false);

    std::vector<NodeId> parent(
        graph.node_count(),
        static_cast<NodeId>(-1));

    std::queue<NodeId> frontier;

    //----------------------------------------------------------

    frontier.push(start);

    visited[start] = true;

    //----------------------------------------------------------

    while (!frontier.empty())
    {
        NodeId current =
            frontier.front();

        frontier.pop();

        result.nodesVisited++;

        //------------------------------------------------------

        if (current == goal)
        {
            result.found = true;
            break;
        }

        //------------------------------------------------------

        for (const Edge& edge :
            graph.neighbors(current))
        {
            if (!visited[edge.to])
            {
                visited[edge.to] = true;

                parent[edge.to] = current;

                frontier.push(edge.to);
            }
        }
    }

    //----------------------------------------------------------
    // Reconstruct path
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