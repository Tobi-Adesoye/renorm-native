#include "renorm/runtime/renorm_optimizer.h"

#include <queue>
#include <vector>

namespace renorm
{

Graph RenormOptimizer::optimize(
    const Graph& graph,
    NodeId start,
    NodeId goal,
    benchmark::BenchmarkMetrics& metrics)
{
    metrics.original_nodes = graph.node_count();
    metrics.original_edges = graph.edge_count();

    //----------------------------------------------------------
    // Reverse graph
    //----------------------------------------------------------

    std::vector<std::vector<NodeId>> reverse(
        graph.node_count());

    for (NodeId u = 0;
         u < graph.node_count();
         ++u)
    {
        for (const auto& edge : graph.neighbors(u))
        {
            reverse[edge.to].push_back(u);
        }
    }

    //----------------------------------------------------------
    // Backward reachability
    //----------------------------------------------------------

    std::vector<bool> live(
        graph.node_count(),
        false);

    std::queue<NodeId> frontier;

    frontier.push(goal);

    live[goal] = true;

    while (!frontier.empty())
    {
        NodeId current =
            frontier.front();

        frontier.pop();

        for (NodeId parent :
             reverse[current])
        {
            if (!live[parent])
            {
                live[parent] = true;

                frontier.push(parent);
            }
        }
    }

    //----------------------------------------------------------
    // Ensure start survives
    //----------------------------------------------------------

    if (!live[start])
    {
        metrics.compiled_nodes = 0;
        metrics.compiled_edges = 0;

        return Graph();
    }

    //----------------------------------------------------------
    // Dense remapping
    //
    // Faster than unordered_map because NodeId is already an
    // integer index.
    //----------------------------------------------------------

    std::vector<NodeId> remap(
        graph.node_count(),
        INVALID_NODE);

    NodeId next = 0;

    for (NodeId i = 0;
         i < graph.node_count();
         ++i)
    {
        if (live[i])
        {
            remap[i] = next++;
        }
    }

    //----------------------------------------------------------
    // Build reduced graph
    //----------------------------------------------------------

    Graph reduced(next);

    for (NodeId u = 0;
         u < graph.node_count();
         ++u)
    {
        if (!live[u])
            continue;

        const NodeId newU =
            remap[u];

        for (const auto& edge :
             graph.neighbors(u))
        {
            if (!live[edge.to])
                continue;

            reduced.add_edge(
                newU,
                remap[edge.to],
                edge.weight);
        }
    }

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    metrics.compiled_nodes =
        reduced.node_count();

    metrics.compiled_edges =
        reduced.edge_count();

    metrics.removed_nodes =
        metrics.original_nodes -
        metrics.compiled_nodes;

    return reduced;
}

} // namespace renorm