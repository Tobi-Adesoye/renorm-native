#include "renorm/benchmark/benchmark_runner.h"

#include "renorm/graph_generator.h"

#include "renorm/search/bfs.h"
#include "renorm/search/dijkstra.h"
#include "renorm/search/astar.h"

#include "renorm/runtime/renorm_runtime.h"

#include "renorm/benchmark/csv_writer.h"
#include "renorm/benchmark/bench_metrics.h"

#include <iostream>
#include <iomanip>
#include <algorithm>

namespace renorm
{
namespace benchmark
{

void BenchmarkRunner::run()
{
    CSVWriter csv;

    if (!csv.open("renorm_benchmark.csv"))
    {
        std::cerr << "Failed to open renorm_benchmark.csv\n";
        return;
    }

    csv.write_header();

    std::cout << "\n";
    std::cout << "=========================================\n";
    std::cout << "RENORM VALIDATION BENCHMARK SUITE\n";
    std::cout << "=========================================\n\n";

    run_graph(
        GraphGenerator::grid(32,32),
        csv,
        "Grid 32x32");

    run_graph(
        GraphGenerator::grid(64,64),
        csv,
        "Grid 64x64");

    run_graph(
        GraphGenerator::grid(128,128),
        csv,
        "Grid 128x128");

    run_graph(
        GraphGenerator::random_sparse(
            5000,
            12000),
        csv,
        "Random Sparse");

    run_graph(
        GraphGenerator::random_dense(
            2000),
        csv,
        "Random Dense");

    run_graph(
        GraphGenerator::scale_free(
            5000),
        csv,
        "Scale Free");

    run_graph(
        GraphGenerator::small_world(
            5000,
            8),
        csv,
        "Small World");

    csv.close();

    std::cout << "\nBenchmark completed.\n";
    std::cout << "CSV written to renorm_benchmark.csv\n";
}

void BenchmarkRunner::run_graph(
    const Graph& graph,
    CSVWriter& csv,
    const std::string& graphName)
{
    std::cout
        << "-----------------------------------------\n";

    std::cout
        << graphName << "\n";

    std::cout
        << "Nodes : "
        << graph.node_count()
        << "\n";

    std::cout
        << "Edges : "
        << graph.edge_count()
        << "\n\n";

    //----------------------------------------------------------
    // BFS
    //----------------------------------------------------------

    BFS bfs;

    SearchResult bfsResult =
        bfs.run(graph);

    BenchMetrics bfsMetrics;

    bfsMetrics.algorithm = "BFS";
    bfsMetrics.graphFamily = graphName;
    bfsMetrics.nodes = graph.node_count();
    bfsMetrics.edges = graph.edge_count();
    bfsMetrics.compileTimeMs = 0.0;
    bfsMetrics.executionTimeMs = bfsResult.executionTimeMs;
    bfsMetrics.totalTimeMs = bfsResult.executionTimeMs;
    bfsMetrics.visitedNodes = bfsResult.nodesVisited;
    bfsMetrics.pathLength = bfsResult.pathLength;

    csv.write_row(bfsMetrics);

    //----------------------------------------------------------
    // Dijkstra
    //----------------------------------------------------------

    Dijkstra dijkstra;

    SearchResult dijkstraResult =
        dijkstra.run(graph);

    BenchMetrics dijkstraMetrics;

    dijkstraMetrics.algorithm = "Dijkstra";
    dijkstraMetrics.graphFamily = graphName;
    dijkstraMetrics.nodes = graph.node_count();
    dijkstraMetrics.edges = graph.edge_count();
    dijkstraMetrics.compileTimeMs = 0.0;
    dijkstraMetrics.executionTimeMs = dijkstraResult.executionTimeMs;
    dijkstraMetrics.totalTimeMs = dijkstraResult.executionTimeMs;
    dijkstraMetrics.visitedNodes = dijkstraResult.nodesVisited;
    dijkstraMetrics.pathLength = dijkstraResult.pathLength;

    csv.write_row(dijkstraMetrics);

    //----------------------------------------------------------
    // A*
    //----------------------------------------------------------

    AStar astar;

    SearchResult astarResult =
        astar.run(graph);

    BenchMetrics astarMetrics;

    astarMetrics.algorithm = "AStar";
    astarMetrics.graphFamily = graphName;
    astarMetrics.nodes = graph.node_count();
    astarMetrics.edges = graph.edge_count();
    astarMetrics.compileTimeMs = 0.0;
    astarMetrics.executionTimeMs = astarResult.executionTimeMs;
    astarMetrics.totalTimeMs = astarResult.executionTimeMs;
    astarMetrics.visitedNodes = astarResult.nodesVisited;
    astarMetrics.pathLength = astarResult.pathLength;

    csv.write_row(astarMetrics);

    //----------------------------------------------------------
    // Renorm
    //----------------------------------------------------------

    runtime::RenormRuntime runtime;

    runtime::RuntimeResult renorm =
        runtime.execute(graph);

    double reduction = 0.0;

    if (graph.node_count() > 0)
    {
        reduction =
            100.0 *
            (
                1.0 -
                static_cast<double>(renorm.nodesVisited) /
                static_cast<double>(graph.node_count())
            );
    }

    BenchMetrics renormMetrics;

    renormMetrics.algorithm = "Renorm";
    renormMetrics.graphFamily = graphName;

    renormMetrics.nodes = graph.node_count();
    renormMetrics.edges = graph.edge_count();

    renormMetrics.compileTimeMs =
        renorm.compileTimeMs;

    renormMetrics.executionTimeMs =
        renorm.executionTimeMs;

    renormMetrics.totalTimeMs =
        renorm.totalTimeMs;

    renormMetrics.visitedNodes =
        renorm.nodesVisited;

    renormMetrics.pathLength =
        renorm.search.pathLength;

    renormMetrics.reductionPercent =
        reduction;

    renormMetrics.floatingPointOperations =
        renorm.floatingPointOperations;

    renormMetrics.blocksExecuted =
        renorm.blocksExecuted;

    //----------------------------------------------------------
    // Derived Metrics
    //----------------------------------------------------------

    if (renorm.totalTimeMs > 0.0)
    {
        renormMetrics.nodesPerSecond =
            static_cast<double>(graph.node_count()) *
            1000.0 /
            renorm.totalTimeMs;

        renormMetrics.edgesPerSecond =
            static_cast<double>(graph.edge_count()) *
            1000.0 /
            renorm.totalTimeMs;
    }

    double fastestBaseline =
        std::min(
            bfsResult.executionTimeMs,
            std::min(
                dijkstraResult.executionTimeMs,
                astarResult.executionTimeMs));

    if (renorm.totalTimeMs > 0.0)
    {
        renormMetrics.speedup =
            fastestBaseline /
            renorm.totalTimeMs;
    }

    //----------------------------------------------------------
    // Reasoning Engine metrics
    //----------------------------------------------------------

    renormMetrics.confidence =
        renorm.confidence;

    renormMetrics.heuristicScore =
        renorm.heuristicScore;

    renormMetrics.frontierQuality =
        renorm.frontierQuality;

    renormMetrics.graphComplexity =
        renorm.graphComplexity;

    renormMetrics.predictionConfidence =
        renorm.predictionConfidence;

    renormMetrics.decisionCost =
        renorm.decisionCost;

    renormMetrics.expansionReduction =
        renorm.expansionReduction;

    renormMetrics.executionPriority =
        renorm.executionPriority;

    renormMetrics.expansionPolicy =
        renorm.expansionPolicy;

    csv.write_row(renormMetrics);

    csv.flush();

    //----------------------------------------------------------
    // Console Output
    //----------------------------------------------------------

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "BFS       : "
        << bfsResult.executionTimeMs
        << " ms\n";

    std::cout
        << "Dijkstra  : "
        << dijkstraResult.executionTimeMs
        << " ms\n";

    std::cout
        << "A* : "
        << astarResult.executionTimeMs
        << " ms\n";

    std::cout
        << "Renorm    : "
        << renorm.totalTimeMs
        << " ms\n";

    std::cout
        << "Reduction : "
        << reduction
        << "%\n\n";
}

}
}