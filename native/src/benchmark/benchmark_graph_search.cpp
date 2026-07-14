#include "renorm/graph.h"

#include "renorm/benchmark_result.h"

#include "renorm/runtime.h"

#include "renorm/compiler.h"

#include <iostream>
#include <iomanip>

int main()
{
    std::cout
        << "RENORM GRAPH BENCHMARK\n\n";

    //------------------------------------------------

    auto graph =
        renorm::GenerateMazeGraph(
            100,
            100);

    //------------------------------------------------

    BenchmarkResult bfs;

    BenchmarkResult dijkstra;

    BenchmarkResult astar;

    BenchmarkResult renorm;

    //------------------------------------------------

    // TODO

    //------------------------------------------------

    std::cout
        << std::setw(12)
        << "Algorithm"

        << std::setw(12)
        << "Nodes"

        << std::setw(12)
        << "Time"

        << std::setw(12)
        << "Solved"

        << "\n";

    //------------------------------------------------

    auto print =
        [](const BenchmarkResult& r)
        {
            std::cout
                << std::setw(12)
                << r.algorithm

                << std::setw(12)
                << r.exploredNodes

                << std::setw(12)
                << r.totalTimeMS

                << std::setw(12)
                << r.solved

                << "\n";
        };

    //------------------------------------------------

    print(bfs);

    print(dijkstra);

    print(astar);

    print(renorm);

    //------------------------------------------------
}