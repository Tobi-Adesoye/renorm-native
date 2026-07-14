#pragma once

#include "renorm/graph.h"

#include <vector>

namespace renorm
{
namespace compiler
{

struct Region
{
    std::size_t id;

    std::vector<NodeId> nodes;
};

class PassRegionDetection
{
public:

    std::vector<Region> run(
        const Graph& graph) const;
};

}
}