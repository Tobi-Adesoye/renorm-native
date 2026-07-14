#include "renorm/graph.h"

#include <algorithm>
#include <stdexcept>

namespace renorm
{

Graph::Graph(std::size_t nodeCount)
    : m_forward(nodeCount),
      m_reverse(nodeCount)
{
}

std::size_t Graph::node_count() const
{
    return m_forward.size();
}

std::size_t Graph::edge_count() const
{
    return m_edgeCount;
}

void Graph::add_edge(
    NodeId from,
    NodeId to,
    float weight)
{
    if (from >= node_count() || to >= node_count())
    {
        throw std::out_of_range(
            "Graph::add_edge() invalid node.");
    }

    Edge edge
    {
        from,
        to,
        weight
    };

    m_forward[from].push_back(edge);
    m_reverse[to].push_back(edge);
    m_edges.push_back(edge);

    ++m_edgeCount;
}

const std::vector<Edge>& Graph::neighbors(
    NodeId node) const
{
    return m_forward[node];
}

const std::vector<Edge>& Graph::predecessors(
    NodeId node) const
{
    return m_reverse[node];
}

bool Graph::has_edge(
    NodeId from,
    NodeId to) const
{
    if (from >= node_count())
    {
        return false;
    }

    const auto& edges = m_forward[from];

    return std::any_of(
        edges.begin(),
        edges.end(),
        [&](const Edge& e)
        {
            return e.to == to;
        });
}

const std::vector<Edge>& Graph::edges() const
{
    return m_edges;
}

Graph Graph::filter_nodes(
    const std::vector<bool>& keep) const
{
    if (keep.size() != node_count())
    {
        throw std::runtime_error(
            "filter_nodes(): size mismatch.");
    }

    std::vector<NodeId> mapping(
        node_count(),
        static_cast<NodeId>(-1));

    NodeId next = 0;

    for (NodeId i = 0; i < node_count(); ++i)
    {
        if (keep[i])
        {
            mapping[i] = next++;
        }
    }

    Graph reduced(next);

    if (keep[m_startNode])
    {
        reduced.set_start_node(
            mapping[m_startNode]);
    }

    if (keep[m_goalNode])
    {
        reduced.set_goal_node(
            mapping[m_goalNode]);
    }

    for (const auto& edge : m_edges)
    {
        if (keep[edge.from] && keep[edge.to])
        {
            reduced.add_edge(
                mapping[edge.from],
                mapping[edge.to],
                edge.weight);
        }
    }

    return reduced;
}

void Graph::clear()
{
    for (auto& edges : m_forward)
    {
        edges.clear();
    }

    for (auto& edges : m_reverse)
    {
        edges.clear();
    }

    m_edges.clear();

    m_edgeCount = 0;
}

void Graph::set_start_node(
    NodeId node)
{
    if (node >= node_count())
    {
        throw std::out_of_range(
            "Invalid start node.");
    }

    m_startNode = node;
}

void Graph::set_goal_node(
    NodeId node)
{
    if (node >= node_count())
    {
        throw std::out_of_range(
            "Invalid goal node.");
    }

    m_goalNode = node;
}

NodeId Graph::start_node() const
{
    return m_startNode;
}

NodeId Graph::goal_node() const
{
    return m_goalNode;
}

} // namespace renorm