#pragma once

#include "renorm/graph.h"

namespace renorm
{
namespace compiler
{

class RenormCompiler
{
public:

    Graph compile(
        const Graph& input);
};

} // namespace compiler
} // namespace renorm