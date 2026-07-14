#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// ParallelBlockExecutor
//
// Executes every compiled block belonging to the current region.
//
// Region
//      ↓
// Block
//      ↓
// Instruction
//
// Phase E executes sequentially.
// Phase F replaces this with true parallel execution.
//--------------------------------------------------------------

class ParallelBlockExecutor
{
public:

    void execute(
        ExecutionContext& context);

private:

    void execute_block(
        ExecutionContext& context,
        std::size_t blockId);

    void execute_instruction(
        ExecutionContext& context,
        const ExecutionInstruction& instruction);
};

}
}