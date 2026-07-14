#include "renorm/search/bfs.h"
#include "renorm/search/dijkstra.h"
#include "renorm/search/astar.h"

#include "renorm/runtime/renorm_runtime.h"

#include "graph_generator.h"

#include <cassert>
#include <iostream>
#include <vector>

using namespace renorm;

namespace
{

void validate_graph(
    const Graph& graph,
    const std::string& name)
{
    //----------------------------------------------------------
    // Algorithms
    //----------------------------------------------------------

    BFS bfs;

    Dijkstra dijkstra;

    AStar astar;

    runtime::RenormRuntime renorm;

    //----------------------------------------------------------
    // Execute
    //----------------------------------------------------------

    SearchResult bfsResult =
        bfs.run(graph);

    SearchResult dijkstraResult =
        dijkstra.run(graph);

    SearchResult astarResult =
        astar.run(graph);

    runtime::RuntimeResult renormResult =
        renorm.execute(graph);

    //----------------------------------------------------------
    // Found
    //----------------------------------------------------------

    assert(
        bfsResult.found ==
        dijkstraResult.found);

    assert(
        bfsResult.found ==
        astarResult.found);

    assert(
        bfsResult.found ==
        renormResult.search.found);

    //----------------------------------------------------------
    // Path Length
    //----------------------------------------------------------

    assert(
        bfsResult.pathLength ==
        dijkstraResult.pathLength);

    assert(
        bfsResult.pathLength ==
        astarResult.pathLength);

    assert(
        bfsResult.pathLength ==
        renormResult.search.pathLength);

    //----------------------------------------------------------
    // Exact Path
    //----------------------------------------------------------

    assert(
        bfsResult.path ==
        dijkstraResult.path);

    assert(
        bfsResult.path ==
        astarResult.path);

    assert(
        bfsResult.path ==
        renormResult.search.path);

    //----------------------------------------------------------
    // Report
    //----------------------------------------------------------

    std::cout
        << "[PASS] "
        << name
        << "\n";

    std::cout
        << "    Nodes Visited : "
        << bfsResult.nodesVisited
        << "\n";

    std::cout
        << "    Path Length   : "
        << bfsResult.pathLength
        << "\n";

    std::cout
        << "    Compile (ms)  : "
        << renormResult.compileTimeMs
        << "\n";

    std::cout
        << "    Runtime (ms)  : "
        << renormResult.executionTimeMs
        << "\n";

    std::cout
        << "    Total (ms)    : "
        << renormResult.totalTimeMs
        << "\n";

    std::cout << "\n";
}

}

int main()
{
    //----------------------------------------------------------
    // Grid
    //----------------------------------------------------------

    validate_graph(
        GraphGenerator::grid(
            64,
            64),
        "Grid");

    //----------------------------------------------------------
    // Dead-End Maze
    //----------------------------------------------------------

    validate_graph(
        GraphGenerator::dead_end_maze(
            64,
            64),
        "Dead-End Maze");

    //----------------------------------------------------------
    // Bottleneck
    //----------------------------------------------------------

    validate_graph(
        GraphGenerator::bottleneck(
            32,
            32),
        "Bottleneck");

    //----------------------------------------------------------
    // Random Sparse
    //----------------------------------------------------------

    validate_graph(
        GraphGenerator::random_sparse(
            1000,
            3000),
        "Random Sparse");

    //----------------------------------------------------------
    // Random Dense
    //----------------------------------------------------------

    validate_graph(
        GraphGenerator::random_dense(
            300),
        "Random Dense");

    //----------------------------------------------------------
    // Scale-Free
    //----------------------------------------------------------

    validate_graph(
        GraphGenerator::scale_free(
            1000),
        "Scale-Free");

    //----------------------------------------------------------
    // Small World
    //----------------------------------------------------------

    validate_graph(
        GraphGenerator::small_world(
            1000,
            6),
        "Small World");

    //----------------------------------------------------------

    std::cout
        << "=========================================\n";

    std::cout
        << "ALL CORRECTNESS TESTS PASSED\n";

    std::cout
        << "=========================================\n";

    return 0;
}