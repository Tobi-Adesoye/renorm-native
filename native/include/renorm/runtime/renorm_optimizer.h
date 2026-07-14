#pragma once

#include "renorm/graph.h"
#include "renorm/benchmark/metrics.h"

namespace renorm
{

class RenormOptimizer
{
public:

    static Graph optimize(
        const Graph& graph,
        NodeId start,
        NodeId goal,
        benchmark::BenchmarkMetrics& metrics);
};

}