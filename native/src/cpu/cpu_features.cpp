#include "renorm/cpu_features.h"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace renorm
{

//==============================================================
// Internal CPU Feature Cache
//==============================================================

static CpuFeatures gFeatures;

static bool gInitialized = false;

//==============================================================
// CPUID Helpers
//==============================================================

static void cpuid(
    int leaf,
    int subleaf,
    int regs[4])
{
#if defined(_MSC_VER)

    __cpuidex(
        regs,
        leaf,
        subleaf);

#else

    __cpuid_count(
        leaf,
        subleaf,
        regs[0],
        regs[1],
        regs[2],
        regs[3]);

#endif
}

//==============================================================
// Detect CPU Features
//==============================================================

static void detect()
{
    if (gInitialized)
    {
        return;
    }

    int regs[4];

    //----------------------------------------------------------
    // Leaf 1
    //----------------------------------------------------------

    cpuid(
        1,
        0,
        regs);

    gFeatures.avx =
        (regs[2] & (1 << 28)) != 0;

    gFeatures.fma =
        (regs[2] & (1 << 12)) != 0;

    //----------------------------------------------------------
    // Leaf 7
    //----------------------------------------------------------

    cpuid(
        7,
        0,
        regs);

    gFeatures.avx2 =
        (regs[1] & (1 << 5)) != 0;

    gFeatures.avx512f =
        (regs[1] & (1 << 16)) != 0;

    gInitialized = true;
}

//==============================================================
// Public Interface
//==============================================================

const CpuFeatures& cpu_features()
{
    detect();

    return gFeatures;
}

//==============================================================
// Convenience Queries
//==============================================================

bool has_avx()
{
    return cpu_features().avx;
}

bool has_fma()
{
    return cpu_features().fma;
}

bool has_avx2()
{
    return cpu_features().avx2;
}

bool has_avx512()
{
    return cpu_features().avx512f;
}

} // namespace renorm