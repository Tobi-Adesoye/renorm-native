#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "renorm/gemm.h"
#include "renorm/kernel_context.h"
#include "renorm/tuner.h" // Added to expose the print_block_config API

using namespace renorm;

int main()
{
    // 1. Diagnostics: Print the dynamic block sizing profile first
    print_block_config();

    constexpr int M = 1024;
    constexpr int K = 1024;
    constexpr int N = 1024;

    KernelContext ctx;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& x : A)
        x = dist(rng);

    for (auto& x : B)
        x = dist(rng);

    auto start = std::chrono::high_resolution_clock::now();

    gemm(
        ctx,
        A.data(),
        B.data(),
        C.data(),
        M,
        N,
        K
    );

    auto end = std::chrono::high_resolution_clock::now();

    double seconds =
        std::chrono::duration<double>(end - start).count();

    double flops =
        2.0 * M * N * K;

    double gflops =
        flops / seconds / 1e9;

    std::cout << "\n=========================================\n";
    std::cout << "RENORM NATIVE GEMM BENCHMARK\n";
    std::cout << "=========================================\n\n";

    std::cout << "Matrix : "
              << M << " x "
              << K << " x "
              << N << "\n\n";

    std::cout << "Elapsed : "
              << seconds
              << " sec\n";

    std::cout << "Performance : "
              << gflops
              << " GFLOPS\n";

    return 0;
}