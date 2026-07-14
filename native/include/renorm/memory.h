#pragma once

#include <cstddef>
#include <cstdint>

namespace renorm {

// ============================================================
// Memory Alignment (cache + SIMD friendly)
// ============================================================

static constexpr std::size_t ALIGNMENT = 64;

// ============================================================
// Aligned Allocation API
// ============================================================

void* aligned_alloc(std::size_t size);

void aligned_free(void* ptr);

// ============================================================
// Utility helpers
// ============================================================

inline bool is_aligned(const void* ptr)
{
    return reinterpret_cast<std::uintptr_t>(ptr) % ALIGNMENT == 0;
}

} // namespace renorm