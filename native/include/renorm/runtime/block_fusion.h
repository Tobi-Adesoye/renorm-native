#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// BlockFusion
//
// Merges compatible execution blocks.
//
// Region
//     ↓
// Block Fusion
//     ↓
// Larger executable block
//
//==============================================================

class BlockFusion
{
public:

    BlockFusion() = default;

    //----------------------------------------------------------

    void fuse(
        ExecutionContext& context);

private:

    bool can_fuse(
        const CompiledBlock& left,
        const CompiledBlock& right) const;

    void merge(
        ExecutionContext& context,
        std::size_t left,
        std::size_t right);
};

}
}