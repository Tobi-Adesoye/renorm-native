#include "pass_reachability.h"

#include <queue>

namespace renorm
{
namespace compiler
{

Graph PassReachability::run(
    const Graph& graph,
    NodeId goalNode) const
{
    //----------------------------------------------------------
    // Reverse Reachability
    //
    // Start from the goal.
    // Walk backwards through predecessors.
    //----------------------------------------------------------

    std::vector<bool> reachable(
        graph.node_count(),
        false);

    std::queue<NodeId> workQueue;

    reachable[goalNode] = true;
    workQueue.push(goalNode);

    while (!workQueue.empty())
    {
        NodeId current = workQueue.front();
        workQueue.pop();

        for (const Edge& edge : graph.predecessors(current))
        {
            NodeId predecessor = edge.from;

            if (!reachable[predecessor])
            {
                reachable[predecessor] = true;
                workQueue.push(predecessor);
            }
        }
    }

    //----------------------------------------------------------
    // Remove unreachable nodes.
    //----------------------------------------------------------

    return graph.filter_nodes(reachable);
}

} // namespace compiler
} // namespace renorm