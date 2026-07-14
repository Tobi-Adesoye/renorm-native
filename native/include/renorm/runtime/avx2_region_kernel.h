#pragma once

#include <cstddef>

#include "renorm/runtime/vector_frontier.h"
#include "renorm/runtime/simd_instruction_packer.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// AVX2RegionKernel
//
// Executes SIMD packets using AVX2.
//
// Responsibilities
//
//   • Load 8 instructions
//   • Gather frontier values
//   • Multiply by weights
//   • Scatter into next frontier
//   • Skip inactive SIMD lanes
//
//==============================================================

class AVX2RegionKernel
{
public:

    AVX2RegionKernel() = default;

    //----------------------------------------------------------
    // Execute complete SIMD instruction block
    //----------------------------------------------------------

    void execute(
        VectorFrontier& frontier,
        const SIMDInstructionBlock& block) const;

private:

    //----------------------------------------------------------
    // Execute one SIMD packet
    //----------------------------------------------------------

    void execute_packet(
        VectorFrontier& frontier,
        const SIMDInstruction& packet) const;
};

} // namespace runtime
} // namespace renorm