#pragma once

#include <vector>

#include "renorm/runtime/vector_frontier.h"
#include "renorm/runtime/simd_block_executor.h"
#include "renorm/runtime/compiled_graph.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// SIMDRegionExecutor
//
// Executes one frontier iteration using SIMD blocks.
//
// Pipeline
//
// Active Regions
//      │
//      ▼
// SIMD Block Executor
//      │
//      ▼
// Vector Frontier
//
//==============================================================

class SIMDRegionExecutor
{
public:

    SIMDRegionExecutor() = default;
    ~SIMDRegionExecutor() = default;

    //----------------------------------------------------------
    // Execute all active regions
    //----------------------------------------------------------

    void execute(
        VectorFrontier& frontier,
        const CompiledGraph& graph,
        const std::vector<std::size_t>& activeRegions) const;

private:

    mutable SIMDBlockExecutor blockExecutor_;
};

}
}