#pragma once

#include "renorm/graph.h"

namespace renorm
{
namespace compiler
{

class PassReachability
{
public:

    Graph run(
        const Graph& graph,
        NodeId goalNode) const;
};

}
}