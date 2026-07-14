#pragma once

#include "renorm/graph.h"

namespace renorm
{
namespace compiler
{

class PassChainCompression
{
public:

    Graph run(const Graph& input) const;
};

}
}