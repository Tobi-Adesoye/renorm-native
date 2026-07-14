#pragma once

namespace renorm
{

struct CpuFeatures
{
    bool avx = false;
    bool fma = false;
    bool avx2 = false;
    bool avx512f = false;
};

const CpuFeatures& cpu_features();

bool has_avx();
bool has_fma();
bool has_avx2();
bool has_avx512();

} // namespace renorm