#pragma once

#include "renorm/graph.h"

#include <vector>
#include <cstddef>

namespace renorm
{

struct SearchResult
{
    //----------------------------------------------------------
    // Search outcome
    //----------------------------------------------------------

    bool found = false;

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    std::size_t nodesVisited = 0;

    std::size_t pathLength = 0;

    double executionTimeMs = 0.0;

    //----------------------------------------------------------
    // Total path cost
    //----------------------------------------------------------

    float pathCost = 0.0f;

    //----------------------------------------------------------
    // Solution path
    //----------------------------------------------------------

    std::vector<NodeId> path;
};

}