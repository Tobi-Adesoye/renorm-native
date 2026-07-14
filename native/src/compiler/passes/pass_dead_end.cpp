#include "pass_dead_end.h"

#include <unordered_set>

namespace renorm
{
namespace compiler
{

static bool leads_to_goal(
    const Graph& graph,
    NodeId node,
    NodeId goalNode,
    std::vector<int>& memo)
{
    if (node == goalNode)
        return true;

    if (memo[node] != -1)
        return memo[node];

    memo[node] = 0; // assume false first (cycle-safe)

    for (const auto& edge : graph.neighbors(node))
    {
        if (leads_to_goal(graph, edge.to, goalNode, memo))
        {
            memo[node] = 1;
            return true;
        }
    }

    return false;
}

Graph PassDeadEnd::run(
    const Graph& graph,
    NodeId goalNode) const
{
    const std::size_t N = graph.node_count();

    std::vector<int> memo(N, -1);
    std::vector<bool> keep(N, false);

    //----------------------------------------------------------
    // Mark all nodes that eventually lead to goal
    //----------------------------------------------------------

    for (NodeId i = 0; i < N; ++i)
    {
        if (leads_to_goal(graph, i, goalNode, memo))
        {
            keep[i] = true;
        }
    }

    //----------------------------------------------------------
    // Ensure goal is always kept
    //----------------------------------------------------------

    keep[goalNode] = true;

    //----------------------------------------------------------
    // Collapse graph
    //----------------------------------------------------------

    return graph.filter_nodes(keep);
}

}
}