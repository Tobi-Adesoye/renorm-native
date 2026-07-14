#pragma once

#include "renorm/runtime/compiled_graph.h"

namespace renorm
{
namespace runtime
{

class GraphCompiler
{
public:

    CompiledGraph compile(
        const Graph& graph);
};

}
}