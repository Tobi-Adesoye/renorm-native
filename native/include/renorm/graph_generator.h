#pragma once

#include "renorm/graph.h"

namespace renorm
{

class GraphGenerator
{
public:

    static Graph grid(
        std::size_t width,
        std::size_t height);

    static Graph dead_end_maze(
        std::size_t width,
        std::size_t height);

    static Graph bottleneck(
        std::size_t branches,
        std::size_t branchLength);

    static Graph random_sparse(
        std::size_t nodes,
        std::size_t edges);

    static Graph random_dense(
        std::size_t nodes);

    static Graph scale_free(
        std::size_t nodes);

    static Graph small_world(
        std::size_t nodes,
        std::size_t degree);
};

}