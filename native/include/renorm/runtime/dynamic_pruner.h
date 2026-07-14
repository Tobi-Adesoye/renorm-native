#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// DynamicPruner
//
// Runtime structural pruning.
//
// Uses:
//
//  • prediction
//  • heuristic
//  • confidence
//  • execution policy
//
// Produces:
//
//  active/inactive regions
//  active/inactive blocks
//  pruning ratio
//
//==============================================================

class DynamicPruner
{
public:

    DynamicPruner() = default;

    //----------------------------------------------------------

    void prune(
        ExecutionContext& context);

private:

    void prune_regions(
        ExecutionContext& context);

    void prune_blocks(
        ExecutionContext& context);

    bool keep_region(
        const ExecutionContext& context,
        const CompiledRegion& region) const;

    bool keep_block(
        const ExecutionContext& context,
        const CompiledBlock& block) const;
};

}
}