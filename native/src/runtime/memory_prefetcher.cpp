#include "renorm/runtime/memory_prefetcher.h"

#include "renorm/runtime/compiled_block.h"
#include "renorm/runtime/execution_instruction.h"

#if defined(__AVX2__) || defined(__SSE2__) || defined(_MSC_VER)
#include <immintrin.h>
#endif

namespace renorm
{
namespace runtime
{

//==============================================================

void MemoryPrefetcher::prefetch_block(
    const CompiledBlock& block) const
{
    if (block.program.empty())
        return;

    //----------------------------------------------------------
    // Prefetch beginning of instruction stream
    //----------------------------------------------------------

    const std::size_t limit =
        block.program.size() <
        PREFETCH_DISTANCE
            ? block.program.size()
            : PREFETCH_DISTANCE;

    for (std::size_t i = 0;
         i < limit;
         ++i)
    {
        prefetch(
            &block.program[i]);
    }
}

//==============================================================

void MemoryPrefetcher::prefetch_instruction(
    const ExecutionInstruction& instruction) const
{
    prefetch(&instruction);
}

//==============================================================

void MemoryPrefetcher::prefetch(
    const void* address)
{
#if defined(__AVX2__) || defined(__SSE2__)

    _mm_prefetch(
        reinterpret_cast<const char*>(address),
        _MM_HINT_T0);

#elif defined(_MSC_VER)

    _mm_prefetch(
        reinterpret_cast<const char*>(address),
        _MM_HINT_T0);

#else

    (void)address;

#endif
}

} // namespace runtime
} // namespace renorm