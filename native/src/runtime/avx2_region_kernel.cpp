#include "renorm/runtime/avx2_region_kernel.h"

#ifdef __AVX2__

#include <immintrin.h>

#endif

namespace renorm
{
namespace runtime
{

//==============================================================

void AVX2RegionKernel::execute(
    VectorFrontier& frontier,
    const SIMDInstructionBlock& block) const
{
    for (const SIMDInstruction& packet :
         block.packets)
    {
        execute_packet(
            frontier,
            packet);
    }
}

//==============================================================

void AVX2RegionKernel::execute_packet(
    VectorFrontier& frontier,
    const SIMDInstruction& packet) const
{
#ifdef __AVX2__

    //----------------------------------------------------------
    // Gather frontier values
    //----------------------------------------------------------

    alignas(32) float sourceValues[8];

    for (std::size_t lane = 0;
         lane < SIMDInstruction::LaneCount;
         ++lane)
    {
        if (packet.activeMask & (1u << lane))
        {
            sourceValues[lane] =
                frontier.current[
                    packet.from[lane]];
        }
        else
        {
            sourceValues[lane] = 0.0f;
        }
    }

    //----------------------------------------------------------
    // SIMD multiply
    //----------------------------------------------------------

    __m256 current =
        _mm256_load_ps(sourceValues);

    __m256 weights =
        _mm256_loadu_ps(packet.weight);

    __m256 result =
        _mm256_mul_ps(
            current,
            weights);

    alignas(32) float destination[8];

    _mm256_store_ps(
        destination,
        result);

    //----------------------------------------------------------
    // Scatter
    //----------------------------------------------------------

    for (std::size_t lane = 0;
         lane < SIMDInstruction::LaneCount;
         ++lane)
    {
        if (!(packet.activeMask & (1u << lane)))
            continue;

        const NodeId node =
            packet.to[lane];

        frontier.next[node] +=
            destination[lane];

        frontier.visited[node] = 1;
    }

#else

    //----------------------------------------------------------
    // Scalar fallback
    //----------------------------------------------------------

    for (std::size_t lane = 0;
         lane < SIMDInstruction::LaneCount;
         ++lane)
    {
        if (!(packet.activeMask & (1u << lane)))
            continue;

        frontier.next[
            packet.to[lane]] +=
            frontier.current[
                packet.from[lane]]
            *
            packet.weight[lane];

        frontier.visited[
            packet.to[lane]] = 1;
    }

#endif
}

} // namespace runtime
} // namespace renorm