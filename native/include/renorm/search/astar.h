#pragma once

#include "renorm/graph.h"
#include "renorm/search/search_result.h"

namespace renorm
{

class AStar
{
public:

    SearchResult run(
        const Graph& graph) const;
};

}