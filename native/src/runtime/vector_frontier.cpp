#include "renorm/runtime/vector_frontier.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

VectorFrontier::VectorFrontier(
    std::size_t nodeCount)
{
    resize(nodeCount);
}

//==============================================================

void VectorFrontier::resize(
    std::size_t nodeCount)
{
    current.assign(nodeCount, 0.0f);

    next.assign(nodeCount, 0.0f);

    visited.assign(nodeCount, 0);

    parent.assign(nodeCount, INVALID_NODE);

    distance.assign(nodeCount, 0.0f);

    depth.assign(nodeCount, 0);

    activeNodes = 0;
}

//==============================================================

void VectorFrontier::clear()
{
    std::fill(
        current.begin(),
        current.end(),
        0.0f);

    std::fill(
        next.begin(),
        next.end(),
        0.0f);

    std::fill(
        visited.begin(),
        visited.end(),
        0);

    std::fill(
        parent.begin(),
        parent.end(),
        INVALID_NODE);

    std::fill(
        distance.begin(),
        distance.end(),
        0.0f);

    std::fill(
        depth.begin(),
        depth.end(),
        0);

    activeNodes = 0;
}

//==============================================================

void VectorFrontier::activate(
    NodeId node)
{
    if (!visited[node])
    {
        visited[node] = 1;

        current[node] = 1.0f;

        ++activeNodes;
    }
}

//==============================================================

bool VectorFrontier::active(
    NodeId node) const
{
    return current[node] != 0.0f;
}

//==============================================================

void VectorFrontier::swap_buffers()
{
    current.swap(next);

    std::fill(
        next.begin(),
        next.end(),
        0.0f);

    activeNodes = 0;

    for (float v : current)
    {
        if (v != 0.0f)
            ++activeNodes;
    }
}

//==============================================================

std::size_t
VectorFrontier::active_count() const
{
    return activeNodes;
}

//==============================================================

std::size_t
VectorFrontier::capacity() const
{
    return current.size();
}

} // namespace runtime
} // namespace renorm