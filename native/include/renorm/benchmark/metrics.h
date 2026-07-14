#pragma once

#include <string>

namespace renorm::benchmark
{

struct BenchmarkMetrics
{
    //----------------------------------------------------------
    // Identity
    //----------------------------------------------------------

    std::string algorithm;

    std::string graph_name;

    //----------------------------------------------------------
    // Correctness
    //----------------------------------------------------------

    bool solved = false;

    int path_length = -1;

    //----------------------------------------------------------
    // Structural Work
    //----------------------------------------------------------

    std::size_t nodes_expanded = 0;

    std::size_t edges_examined = 0;

    //----------------------------------------------------------
    // Renorm Compiler Metrics
    //----------------------------------------------------------

    std::size_t original_nodes = 0;

    std::size_t compiled_nodes = 0;

    std::size_t removed_nodes = 0;

    std::size_t original_edges = 0;

    std::size_t compiled_edges = 0;

    //----------------------------------------------------------
    // Runtime Metrics
    //----------------------------------------------------------

    std::size_t frontier_updates = 0;

    std::size_t matrix_operations = 0;

    std::size_t gemm_calls = 0;

    //----------------------------------------------------------
    // Timing
    //----------------------------------------------------------

    double compile_time_ms = 0.0;

    double execution_time_ms = 0.0;

    double total_time_ms = 0.0;
};

}