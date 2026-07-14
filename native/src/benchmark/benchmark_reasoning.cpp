#include "renorm/runtime/reasoning/reasoning_engine.h"

#include "renorm/runtime/frontier.h"
#include "renorm/runtime/runtime_result.h"
#include "renorm/runtime/compiler/graph_compiler.h"

#include "renorm/graph_generator.h"

#include <chrono>
#include <iostream>
#include <iomanip>

using namespace renorm;
using namespace renorm::runtime;

int main()
{
    std::cout << "\n";
    std::cout << "=========================================\n";
    std::cout << "RENORM REASONING ENGINE BENCHMARK\n";
    std::cout << "=========================================\n\n";

    //----------------------------------------------------------
    // Generate benchmark graph
    //----------------------------------------------------------

    Graph graph =
        GraphGenerator::grid(256,256);

    //----------------------------------------------------------
    // Compile graph
    //----------------------------------------------------------

    GraphCompiler compiler;

    CompiledGraph compiled =
        compiler.compile(graph);

    //----------------------------------------------------------
    // Frontier
    //----------------------------------------------------------

    Frontier frontier(graph.node_count());

    frontier.startNode = 0;

    frontier.goalNode =
        graph.node_count() - 1;

    frontier.active.push_back(frontier.startNode);

    frontier.activeNodeCount = 1;

    frontier.nodesVisited = 1;

    //----------------------------------------------------------
    // Runtime result
    //----------------------------------------------------------

    RuntimeResult result;

    //----------------------------------------------------------
    // Reasoning engine
    //----------------------------------------------------------

    ReasoningEngine engine;

    //----------------------------------------------------------
    // Benchmark
    //----------------------------------------------------------

    auto t0 =
        std::chrono::high_resolution_clock::now();

    engine.execute(
        frontier,
        compiled,
        result);

    auto t1 =
        std::chrono::high_resolution_clock::now();

    double elapsed =
        std::chrono::duration<double,std::milli>(
            t1-t0).count();

    //----------------------------------------------------------
    // Output
    //----------------------------------------------------------

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "Reasoning Time        : "
        << elapsed
        << " ms\n";

    std::cout
        << "Confidence            : "
        << result.confidence
        << "\n";

    std::cout
        << "Prediction Accuracy   : "
        << result.predictionAccuracy
        << "\n";

    std::cout
        << "Decision Cost         : "
        << result.decisionCost
        << "\n";

    std::cout
        << "Expansion Reduction   : "
        << result.expansionReduction
        << "%\n";

    std::cout
        << "Predicted Nodes       : "
        << result.predictedVisited
        << "\n";

    std::cout << "\nBenchmark Complete\n";

    return 0;
}