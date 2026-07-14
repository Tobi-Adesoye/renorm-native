#pragma once

#include "renorm/graph.h"

namespace renorm
{
namespace compiler
{

class PassEquivalenceFolding
{
public:

    Graph run(const Graph& input) const;
};

}
}