#include "renorm/tuner.h"
#include "renorm/cpu_features.h"
#include <algorithm>
#include <iostream>

namespace renorm
{

BlockConfig tune_for_cpu()
{
    BlockConfig cfg;
    const auto& cpu = cpu_features();

    // Mapping based strictly on existing structural profile fields
    if (cpu.avx512f)
    {
        cfg.MC = 128;
        cfg.NC = 256;
        cfg.KC = 256;
    }
    else if (cpu.avx2)
    {
        cfg.MC = 64;
        cfg.NC = 256;
        cfg.KC = 128;
    }
    else
    {
        cfg.MC = 32;
        cfg.NC = 64;
        cfg.KC = 64;
    }

    // Invariant Constraints: Block limits must remain exact multiples of MR/NR (8)
    cfg.MC = std::max(32, (cfg.MC / 8) * 8);
    cfg.NC = std::max(32, (cfg.NC / 8) * 8);
    cfg.KC = std::max(32, (cfg.KC / 8) * 8);

    return cfg;
}

const BlockConfig& get_block_config()
{
    static const BlockConfig cached_config = tune_for_cpu();
    return cached_config;
}

void print_block_config()
{
    const auto& cpu = cpu_features();
    const auto& cfg = get_block_config();

    std::cout << "\n========================================\n";
    std::cout << "        RENORM Hardware Profile        \n";
    std::cout << "========================================\n";
    std::cout << "AVX-512 Supported : " << (cpu.avx512f ? "Yes" : "No") << "\n";
    std::cout << "AVX2 Supported    : " << (cpu.avx2 ? "Yes" : "No") << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Tuned MC Block    : " << cfg.MC << "\n";
    std::cout << "Tuned NC Block    : " << cfg.NC << "\n";
    std::cout << "Tuned KC Block    : " << cfg.KC << "\n";
    std::cout << "========================================\n\n";
}

} // namespace renorm