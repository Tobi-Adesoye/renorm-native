#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "renorm/runtime/compiled_block.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// SIMDInstruction
//
// One SIMD packet.
//
// Designed for AVX2 (8 float lanes).
//
//==============================================================

struct SIMDInstruction
{
    static constexpr std::size_t LaneCount = 8;

    NodeId from[LaneCount];

    NodeId to[LaneCount];

    float weight[LaneCount];

    std::uint8_t activeMask = 0;
};

//==============================================================
//
// SIMDInstructionBlock
//
// Vectorized instruction stream.
//
//==============================================================

struct SIMDInstructionBlock
{
    std::vector<SIMDInstruction> packets;

    std::size_t instructionCount = 0;

    std::size_t packetCount = 0;

    void clear();

    void reserve(std::size_t packetsRequired);
};

//==============================================================
//
// SIMDInstructionPacker
//
// Converts scalar instruction streams into SIMD packets.
//
//==============================================================

class SIMDInstructionPacker
{
public:

    SIMDInstructionPacker() = default;

    SIMDInstructionBlock pack(
        const CompiledBlock& block) const;
};

} // namespace runtime
} // namespace renorm