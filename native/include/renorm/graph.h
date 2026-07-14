#pragma once

#include <cstddef>
#include <vector>
#include <limits>

namespace renorm
{

//--------------------------------------------------------------
// Node Types
//--------------------------------------------------------------

using NodeId = std::size_t;

constexpr NodeId INVALID_NODE =
    std::numeric_limits<NodeId>::max();


//--------------------------------------------------------------
// Directed Edge
//--------------------------------------------------------------

struct Edge
{
    NodeId from = INVALID_NODE;
    NodeId to = INVALID_NODE;
    float weight = 1.0f;
};


//--------------------------------------------------------------
// Graph
//--------------------------------------------------------------

class Graph
{
public:

    explicit Graph(std::size_t nodeCount = 0);

    //----------------------------------------------------------
    // Graph Size
    //----------------------------------------------------------

    std::size_t node_count() const;

    std::size_t edge_count() const;

    //----------------------------------------------------------
    // Graph Construction
    //----------------------------------------------------------

    void add_edge(
        NodeId from,
        NodeId to,
        float weight = 1.0f);

    void clear();

    //----------------------------------------------------------
    // Adjacency
    //----------------------------------------------------------

    const std::vector<Edge>& neighbors(
        NodeId node) const;

    const std::vector<Edge>& predecessors(
        NodeId node) const;

    bool has_edge(
        NodeId from,
        NodeId to) const;

    //----------------------------------------------------------
    // Complete Edge List
    //----------------------------------------------------------

    const std::vector<Edge>& edges() const;

    //----------------------------------------------------------
    // Graph Reduction
    //----------------------------------------------------------

    Graph filter_nodes(
        const std::vector<bool>& keep) const;

    //----------------------------------------------------------
    // Search Metadata
    //----------------------------------------------------------

    void set_start_node(
        NodeId node);

    void set_goal_node(
        NodeId node);

    NodeId start_node() const;

    NodeId goal_node() const;

private:

    //----------------------------------------------------------
    // Forward adjacency
    //----------------------------------------------------------

    std::vector<std::vector<Edge>> m_forward;

    //----------------------------------------------------------
    // Reverse adjacency
    //----------------------------------------------------------

    std::vector<std::vector<Edge>> m_reverse;

    //----------------------------------------------------------
    // Complete edge list
    //----------------------------------------------------------

    std::vector<Edge> m_edges;

    //----------------------------------------------------------
    // Metadata
    //----------------------------------------------------------

    std::size_t m_edgeCount = 0;

    NodeId m_startNode = INVALID_NODE;

    NodeId m_goalNode = INVALID_NODE;
};

} // namespace renorm