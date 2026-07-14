#pragma once

#include "renorm/graph.h"
#include "renorm/runtime/simd_instruction_packer.h"

#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// ExecutionInstruction
//
// Scalar instruction emitted by compiler.
//
//==============================================================

struct ExecutionInstruction
{
    NodeId from = INVALID_NODE;

    NodeId to = INVALID_NODE;

    float weight = 0.0f;
};

//==============================================================
//
// CompiledBlock
//
// Runtime executable block.
//
// Compiler emits:
//
//      Scalar Program
//              ↓
//      SIMD Program
//
// Runtime chooses:
//
//      Scalar Executor
//              or
//      SIMD Executor
//
//==============================================================

struct CompiledBlock
{
    //----------------------------------------------------------
    // Runtime identifier
    //----------------------------------------------------------

    std::size_t id = 0;

    //----------------------------------------------------------
    // Nodes
    //----------------------------------------------------------

    std::vector<NodeId> nodes;

    //----------------------------------------------------------
    // Scalar instruction stream
    //----------------------------------------------------------

    std::vector<ExecutionInstruction> program;

    //----------------------------------------------------------
    // SIMD instruction stream
    //
    // Produced once by compiler.
    //----------------------------------------------------------

    SIMDInstructionProgram simdProgram;

    //----------------------------------------------------------
    // Dependencies
    //----------------------------------------------------------

    std::vector<std::size_t> outgoingBlocks;

    std::vector<std::size_t> incomingBlocks;

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    std::size_t instructionCount = 0;

    std::size_t simdBatchCount = 0;

    std::size_t edgeCount = 0;

    float density = 0.0f;

    //----------------------------------------------------------
    // Reserve helpers
    //----------------------------------------------------------

    inline void reserve_nodes(
        std::size_t count)
    {
        nodes.reserve(count);
    }

    inline void reserve_program(
        std::size_t count)
    {
        program.reserve(count);
    }

    //----------------------------------------------------------
    // Add scalar instruction
    //----------------------------------------------------------

    inline void add_instruction(
        NodeId from,
        NodeId to,
        float weight)
    {
        program.push_back(
        {
            from,
            to,
            weight
        });

        instructionCount =
            program.size();

        edgeCount =
            instructionCount;
    }

    //----------------------------------------------------------
    // Install SIMD program
    //----------------------------------------------------------

    inline void install_simd_program(
        SIMDInstructionProgram&& packed)
    {
        simdProgram =
            std::move(packed);

        simdBatchCount =
            simdProgram.batches.size();
    }

    //----------------------------------------------------------
    // Helpers
    //----------------------------------------------------------

    inline bool has_simd() const
    {
        return !simdProgram.batches.empty();
    }

    inline std::size_t scalar_instruction_count() const
    {
        return program.size();
    }

    inline std::size_t simd_batch_count() const
    {
        return simdProgram.batches.size();
    }

    inline void clear()
    {
        nodes.clear();

        program.clear();

        simdProgram.clear();

        outgoingBlocks.clear();

        incomingBlocks.clear();

        instructionCount = 0;

        simdBatchCount = 0;

        edgeCount = 0;

        density = 0.0f;
    }
};

} // namespace runtime
} // namespace renorm