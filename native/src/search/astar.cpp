#include "renorm/search/astar.h"

#include <queue>
#include <vector>
#include <limits>
#include <chrono>
#include <algorithm>

namespace renorm
{

namespace
{

//----------------------------------------------------------
// Heuristic
//
// Currently returns zero because Graph has no spatial
// coordinates.
//
// Later this becomes:
//
// Manhattan
// Euclidean
// Octile
//
// etc.
//----------------------------------------------------------

inline float heuristic(
    NodeId,
    NodeId)
{
    return 0.0f;
}

struct AStarNode
{
    float fScore;

    float gScore;

    NodeId node;

    bool operator>(const AStarNode& other) const
    {
        return fScore > other.fScore;
    }
};

}

//==========================================================

SearchResult AStar::run(
    const Graph& graph) const
{
    SearchResult result;

    if (graph.node_count() == 0)
    {
        return result;
    }

    const NodeId start = 0;

    const NodeId goal =
        graph.node_count() - 1;

    auto begin =
        std::chrono::high_resolution_clock::now();

    //------------------------------------------------------

    const float INF =
        std::numeric_limits<float>::infinity();

    std::vector<float> gScore(
        graph.node_count(),
        INF);

    std::vector<bool> visited(
        graph.node_count(),
        false);

    std::vector<NodeId> parent(
        graph.node_count(),
        static_cast<NodeId>(-1));

    std::priority_queue<
        AStarNode,
        std::vector<AStarNode>,
        std::greater<AStarNode>> frontier;

    //------------------------------------------------------

    gScore[start] = 0.0f;

    frontier.push(
    {
        heuristic(start, goal),
        0.0f,
        start
    });

    //------------------------------------------------------

    while (!frontier.empty())
    {
        AStarNode current =
            frontier.top();

        frontier.pop();

        if (visited[current.node])
        {
            continue;
        }

        visited[current.node] = true;

        result.nodesVisited++;

        if (current.node == goal)
        {
            result.found = true;
            break;
        }

        //--------------------------------------------------

        for (const Edge& edge :
            graph.neighbors(current.node))
        {
            float tentativeG =
                gScore[current.node] +
                edge.weight;

            if (tentativeG < gScore[edge.to])
            {
                gScore[edge.to] =
                    tentativeG;

                parent[edge.to] =
                    current.node;

                float f =
                    tentativeG +
                    heuristic(edge.to, goal);

                frontier.push(
                {
                    f,
                    tentativeG,
                    edge.to
                });
            }
        }
    }

    //------------------------------------------------------
    // Path reconstruction
    //------------------------------------------------------

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

    //------------------------------------------------------

    auto end =
        std::chrono::high_resolution_clock::now();

    result.executionTimeMs =
        std::chrono::duration<double,std::milli>(
            end - begin).count();

    return result;
}

}