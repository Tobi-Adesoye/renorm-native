#include "renorm/runtime/simd_block_executor.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void SIMDBlockExecutor::execute(
    VectorFrontier& frontier,
    const CompiledBlock& block) const
{
    //----------------------------------------------------------
    // Convert scalar instruction stream
    // into SIMD packets.
    //----------------------------------------------------------

    SIMDInstructionBlock simdBlock =
        packer_.pack(block);

    //----------------------------------------------------------
    // Execute packets
    //----------------------------------------------------------

    kernel_.execute(
        frontier,
        simdBlock);
}

} // namespace runtime
} // namespace renorm