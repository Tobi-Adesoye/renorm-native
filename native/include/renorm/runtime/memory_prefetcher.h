#pragma once

#include <cstddef>

namespace renorm
{
namespace runtime
{

class CompiledBlock;
class ExecutionInstruction;

//==============================================================
//
// MemoryPrefetcher
//
// Inserts architecture-specific cache prefetches during
// execution.
//
// Goals
//
// • Hide DRAM latency
// • Warm L1/L2 cache
// • Improve SIMD throughput
// • Reduce memory stalls
//
//==============================================================

class MemoryPrefetcher
{
public:

    MemoryPrefetcher() = default;

    //----------------------------------------------------------
    // Prefetch entire block
    //----------------------------------------------------------

    void prefetch_block(
        const CompiledBlock& block) const;

    //----------------------------------------------------------
    // Prefetch instruction
    //----------------------------------------------------------

    void prefetch_instruction(
        const ExecutionInstruction& instruction) const;

    //----------------------------------------------------------
    // Prefetch arbitrary address
    //----------------------------------------------------------

    static void prefetch(
        const void* address);

private:

    static constexpr std::size_t PREFETCH_DISTANCE = 16;
};

} // namespace runtime
} // namespace renorm