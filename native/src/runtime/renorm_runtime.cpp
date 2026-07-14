#include "renorm/runtime/renorm_runtime.h"

#include "renorm/runtime/graph_compiler.h"
#include "renorm/runtime/frontier_builder.h"
#include "renorm/runtime/runtime_execution_engine.h"

#include <chrono>

namespace renorm
{
namespace runtime
{

//==============================================================
// Convenience
//==============================================================

RuntimeResult RenormRuntime::execute(
    const Graph& graph)
{
    return execute(
        graph,
        0,
        graph.node_count() > 0
            ? graph.node_count() - 1
            : 0);
}

//==============================================================

RuntimeResult RenormRuntime::execute(
    const CompiledGraph& compiled)
{
    return execute(
        compiled,
        0,
        compiled.nodeCount > 0
            ? compiled.nodeCount - 1
            : 0);
}

//==============================================================
// Compile + Execute
//==============================================================

RuntimeResult RenormRuntime::execute(
    const Graph& graph,
    NodeId start,
    NodeId goal)
{
    auto compileBegin =
        std::chrono::high_resolution_clock::now();

    GraphCompiler compiler;

    CompiledGraph compiled =
        compiler.compile(graph);

    auto compileEnd =
        std::chrono::high_resolution_clock::now();

    RuntimeResult result =
        execute(
            compiled,
            start,
            goal);

    result.compileTimeMs =
        std::chrono::duration<double,std::milli>(
            compileEnd - compileBegin).count();

    result.totalTimeMs =
        result.compileTimeMs +
        result.executionTimeMs;

    return result;
}

//==============================================================
// Execute compiled graph
//==============================================================

RuntimeResult RenormRuntime::execute(
    const CompiledGraph& compiled,
    NodeId start,
    NodeId goal)
{
    //----------------------------------------------------------
    // Build execution context
    //----------------------------------------------------------

    ExecutionContext context;

    context.attach(compiled);

    //----------------------------------------------------------
    // Build runtime frontier
    //----------------------------------------------------------

    FrontierBuilder frontierBuilder;

    context.frontier =
        frontierBuilder.build(
            compiled,
            start,
            goal);

    //----------------------------------------------------------
    // Region frontier
    //----------------------------------------------------------

    context.regionFrontier.build(
        compiled,
        0);

    //----------------------------------------------------------
    // Runtime timer
    //----------------------------------------------------------

    auto runtimeBegin =
        std::chrono::high_resolution_clock::now();

    //----------------------------------------------------------
    // Execute runtime pipeline
    //----------------------------------------------------------

    RuntimeExecutionEngine engine;

    engine.execute(context);

    //----------------------------------------------------------
    // Timing
    //----------------------------------------------------------

    auto runtimeEnd =
        std::chrono::high_resolution_clock::now();

    context.result.executionTimeMs =
        std::chrono::duration<double,std::milli>(
            runtimeEnd - runtimeBegin).count();

    context.result.totalTimeMs =
        context.result.compileTimeMs +
        context.result.executionTimeMs;

    //----------------------------------------------------------
    // Search metadata
    //----------------------------------------------------------

    context.result.search.executionTimeMs =
        context.result.executionTimeMs;

    return context.result;
}

} // namespace runtime
} // namespace renorm