#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// InstructionFusion
//
// Performs runtime instruction fusion.
//
// Goal:
//
//   Reduce instruction dispatch overhead by merging adjacent
//   compatible instructions inside CompiledBlocks.
//
//==============================================================

class InstructionFusion
{
public:

    InstructionFusion() = default;

    //----------------------------------------------------------

    void fuse(
        ExecutionContext& context);

private:

    //----------------------------------------------------------

    void fuse_block(
        ExecutionContext& context,
        std::size_t blockId);

    bool can_fuse(
        const ExecutionContext& context,
        std::size_t leftInstruction,
        std::size_t rightInstruction) const;

    void merge(
        ExecutionContext& context,
        std::size_t blockId,
        std::size_t leftInstruction,
        std::size_t rightInstruction);
};

}
}