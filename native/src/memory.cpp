#include "renorm/memory.h"

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace renorm {

// ============================================================
// Aligned Allocation
// ============================================================

void* aligned_alloc(std::size_t size)
{
#if defined(_MSC_VER)
    return _aligned_malloc(size, ALIGNMENT);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGNMENT, size) != 0)
    {
        return nullptr;
    }
    return ptr;
#endif
}

// ============================================================
// Free aligned memory
// ============================================================

void aligned_free(void* ptr)
{
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace renorm