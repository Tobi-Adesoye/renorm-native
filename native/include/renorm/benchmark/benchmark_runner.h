#pragma once

#include "renorm/graph.h"
#include "renorm/benchmark/csv_writer.h"

#include <string>

namespace renorm
{
namespace benchmark
{

class BenchmarkRunner
{
public:

    void run();

private:

    void run_graph(
        const Graph& graph,
        benchmark::CSVWriter& csv,
        const std::string& graphName);
};

}
}