#pragma once

#include "renorm/runtime/frontier_executor.h"

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// FrontierExecutorGEMM
//
// Blocked GEMM-based frontier executor.
//
// Extends FrontierExecutor with a specialized execution path
// that processes graph blocks using the compiled blocked matrix.
//--------------------------------------------------------------

class FrontierExecutorGEMM : public FrontierExecutor
{
public:

    FrontierExecutorGEMM() = default;
    ~FrontierExecutorGEMM() override = default;

    //----------------------------------------------------------
    // Execute complete runtime until completion
    //----------------------------------------------------------

    void execute(
        Frontier& frontier,
        const CompiledGraph& compiled);

protected:

    //----------------------------------------------------------
    // Execute one frontier iteration
    //----------------------------------------------------------

    void execute_iteration(
        Frontier& frontier,
        const CompiledGraph& compiled) override;
};

} // namespace runtime
} // namespace renorm