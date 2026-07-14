#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// BlockScheduler
//
// Selects executable blocks inside the current region.
//
// Region
//    ↓
// Active Blocks
//    ↓
// Executor
//--------------------------------------------------------------

class BlockScheduler
{
public:

    BlockScheduler() = default;

    //----------------------------------------------------------
    // Build block queue for current region
    //----------------------------------------------------------

    void schedule(
        ExecutionContext& context);

private:

    bool should_execute(
        const ExecutionContext& context,
        std::size_t blockId) const;

    float compute_priority(
        const ExecutionContext& context,
        std::size_t blockId) const;
};

}
}