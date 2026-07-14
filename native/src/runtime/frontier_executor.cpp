#include "renorm/runtime/frontier_executor.h"

#include "renorm/runtime/frontier_executor_gemm.h"

namespace renorm
{
namespace runtime
{

void FrontierExecutor::execute(
    Frontier& frontier,
    const CompiledGraph& compiled)
{
    //----------------------------------------------------------
    // Phase B2
    //
    // Delegate execution to the compiled execution engine.
    //----------------------------------------------------------

    FrontierExecutorGEMM executor;

    executor.execute(
        frontier,
        compiled);
}

} // namespace runtime
} // namespace renorm