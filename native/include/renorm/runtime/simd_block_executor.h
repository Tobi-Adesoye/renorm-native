#pragma once

#include "renorm/runtime/compiled_graph.h"
#include "renorm/runtime/vector_frontier.h"
#include "renorm/runtime/simd_instruction_packer.h"
#include "renorm/runtime/avx2_region_kernel.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// SIMDBlockExecutor
//
// Executes one compiled block using SIMD.
//
// Pipeline
//
// CompiledBlock
//      │
//      ▼
// SIMDInstructionPacker
//      │
//      ▼
// SIMDInstructionBlock
//      │
//      ▼
// AVX2RegionKernel
//
//==============================================================

class SIMDBlockExecutor
{
public:

    SIMDBlockExecutor() = default;

    //----------------------------------------------------------
    // Execute one compiled block
    //----------------------------------------------------------

    void execute(
        VectorFrontier& frontier,
        const CompiledBlock& block) const;

private:

    mutable SIMDInstructionPacker packer_;

    mutable AVX2RegionKernel kernel_;
};

} // namespace runtime
} // namespace renorm