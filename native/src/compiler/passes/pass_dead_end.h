#pragma once

#include "renorm/graph.h"

namespace renorm
{
namespace compiler
{

class PassDeadEnd
{
public:

    Graph run(
        const Graph& graph,
        NodeId goalNode) const;
};

}
}