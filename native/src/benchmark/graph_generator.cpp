#include "renorm/graph_generator.h"

#include <random>
#include <unordered_set>
#include <algorithm>

namespace renorm
{

namespace
{

std::mt19937 rng(123456);

}

//==============================================================
// Grid
//==============================================================

Graph GraphGenerator::grid(
    std::size_t width,
    std::size_t height)
{
    const std::size_t nodes =
        width * height;

    Graph graph(nodes);

    auto id =
        [width](std::size_t x,
                std::size_t y)
    {
        return y * width + x;
    };

    for (std::size_t y = 0; y < height; ++y)
    {
        for (std::size_t x = 0; x < width; ++x)
        {
            NodeId current =
                id(x, y);

            if (x + 1 < width)
            {
                graph.add_edge(
                    current,
                    id(x + 1, y));

                graph.add_edge(
                    id(x + 1, y),
                    current);
            }

            if (y + 1 < height)
            {
                graph.add_edge(
                    current,
                    id(x, y + 1));

                graph.add_edge(
                    id(x, y + 1),
                    current);
            }
        }
    }

    graph.set_start_node(0);
    graph.set_goal_node(nodes - 1);

    return graph;
}

//==============================================================
// Dead-end Maze
//==============================================================

Graph GraphGenerator::dead_end_maze(
    std::size_t width,
    std::size_t height)
{
    Graph graph =
        grid(width, height);

    std::uniform_int_distribution<int>
        removeChance(0,3);

    for (std::size_t y = 1; y < height - 1; ++y)
    {
        for (std::size_t x = 1; x < width - 1; ++x)
        {
            if (removeChance(rng) == 0)
            {
                NodeId n =
                    y * width + x;

                // create small dead-end branch

                graph.add_edge(
                    n,
                    n);

            }
        }
    }

    return graph;
}

//==============================================================
// Bottleneck
//==============================================================

Graph GraphGenerator::bottleneck(
    std::size_t branches,
    std::size_t branchLength)
{
    std::size_t nodes =
        branches * branchLength + 2;

    Graph graph(nodes);

    NodeId start = 0;
    NodeId goal = nodes - 1;

    NodeId current = 1;

    for (std::size_t b = 0; b < branches; ++b)
    {
        graph.add_edge(
            start,
            current);

        NodeId previous =
            current;

        for (std::size_t i = 1;
             i < branchLength;
             ++i)
        {
            graph.add_edge(
                previous,
                current + 1);

            previous =
                current + 1;

            current++;
        }

        graph.add_edge(
            previous,
            goal);

        current++;
    }

    graph.set_start_node(start);
    graph.set_goal_node(goal);

    return graph;
}

//==============================================================
// Random Sparse
//==============================================================

Graph GraphGenerator::random_sparse(
    std::size_t nodes,
    std::size_t edges)
{
    Graph graph(nodes);

    std::uniform_int_distribution<NodeId>
        pick(0, nodes - 1);

    while (graph.edge_count() < edges)
    {
        NodeId u = pick(rng);
        NodeId v = pick(rng);

        if (u == v)
            continue;

        if (graph.has_edge(u,v))
            continue;

        graph.add_edge(u,v);
    }

    graph.set_start_node(0);
    graph.set_goal_node(nodes - 1);

    return graph;
}

//==============================================================
// Random Dense
//==============================================================

Graph GraphGenerator::random_dense(
    std::size_t nodes)
{
    Graph graph(nodes);

    for (NodeId i=0;i<nodes;i++)
    {
        for (NodeId j=0;j<nodes;j++)
        {
            if (i==j)
                continue;

            graph.add_edge(i,j);
        }
    }

    graph.set_start_node(0);
    graph.set_goal_node(nodes-1);

    return graph;
}

//==============================================================
// Scale-Free
//==============================================================

Graph GraphGenerator::scale_free(
    std::size_t nodes)
{
    Graph graph(nodes);

    if(nodes<2)
        return graph;

    graph.add_edge(0,1);

    std::vector<NodeId> degreeList =
    {
        0,1
    };

    std::uniform_int_distribution<int> coin(0,1);

    for(NodeId n=2;n<nodes;n++)
    {
        std::uniform_int_distribution<std::size_t>
            pick(0,degreeList.size()-1);

        NodeId target =
            degreeList[pick(rng)];

        graph.add_edge(n,target);

        if(coin(rng))
            graph.add_edge(target,n);

        degreeList.push_back(target);
        degreeList.push_back(n);
    }

    graph.set_start_node(0);
    graph.set_goal_node(nodes-1);

    return graph;
}

//==============================================================
// Small World
//==============================================================

Graph GraphGenerator::small_world(
    std::size_t nodes,
    std::size_t degree)
{
    Graph graph(nodes);

    std::uniform_int_distribution<NodeId>
        pick(0,nodes-1);

    for(NodeId i=0;i<nodes;i++)
    {
        for(std::size_t d=1;d<=degree;d++)
        {
            NodeId j =
                (i+d)%nodes;

            graph.add_edge(i,j);
            graph.add_edge(j,i);
        }
    }

    std::uniform_real_distribution<float>
        probability(0.0f,1.0f);

    for(NodeId i=0;i<nodes;i++)
    {
        if(probability(rng)<0.1f)
        {
            NodeId randomNode =
                pick(rng);

            if(randomNode!=i)
                graph.add_edge(i,randomNode);
        }
    }

    graph.set_start_node(0);
    graph.set_goal_node(nodes-1);

    return graph;
}

}