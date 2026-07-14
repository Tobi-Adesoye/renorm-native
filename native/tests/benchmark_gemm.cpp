#include "renorm/gemm.h"
#include "renorm/kernel_context.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static void fill_random(std::vector<float>& v)
{
    static std::mt19937 rng(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& x : v)
        x = dist(rng);
}

struct BenchmarkResult
{
    int M;
    int N;
    int K;

    double best_ms;
    double avg_ms;
    double worst_ms;

    double gflops;
};

static BenchmarkResult benchmark(
    int M,
    int N,
    int K,
    int iterations = 10)
{
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);

    fill_random(A);
    fill_random(B);

    renorm::KernelContext ctx;

    //----------------------------------------------------------
    // Warmup
    //----------------------------------------------------------

    renorm::gemm(
        ctx,
        A.data(),
        B.data(),
        C.data(),
        M,
        N,
        K);

    //----------------------------------------------------------
    // Timed Runs
    //----------------------------------------------------------

    std::vector<double> times;

    for (int i = 0; i < iterations; ++i)
    {
        std::fill(C.begin(), C.end(), 0.0f);

        auto start = Clock::now();

        renorm::gemm(
            ctx,
            A.data(),
            B.data(),
            C.data(),
            M,
            N,
            K);

        auto end = Clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(
                end - start).count();

        times.push_back(ms);
    }

    auto minmax =
        std::minmax_element(
            times.begin(),
            times.end());

    double total = 0.0;

    for (double t : times)
        total += t;

    double avg =
        total / times.size();

    double seconds =
        (*minmax.first) / 1000.0;

    double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    double gflops =
        flops /
        seconds /
        1e9;

    return
    {
        M,
        N,
        K,
        *minmax.first,
        avg,
        *minmax.second,
        gflops
    };
}

int main()
{
    std::vector<int> sizes =
    {
        64,
        128,
        256,
        512,
        1024,
        2048
    };

    std::ofstream csv("benchmark_results.csv");

    csv
        << "M,N,K,"
        << "Best(ms),"
        << "Average(ms),"
        << "Worst(ms),"
        << "GFLOPS\n";

    std::cout
        << std::left
        << std::setw(8)  << "M"
        << std::setw(8)  << "N"
        << std::setw(8)  << "K"
        << std::setw(14) << "Best(ms)"
        << std::setw(14) << "Avg(ms)"
        << std::setw(14) << "Worst(ms)"
        << std::setw(14) << "GFLOPS"
        << "\n";

    std::cout
        << std::string(80, '-')
        << "\n";

    for (int s : sizes)
    {
        auto r =
            benchmark(
                s,
                s,
                s);

        std::cout
            << std::setw(8) << r.M
            << std::setw(8) << r.N
            << std::setw(8) << r.K
            << std::setw(14) << std::fixed << std::setprecision(3) << r.best_ms
            << std::setw(14) << r.avg_ms
            << std::setw(14) << r.worst_ms
            << std::setw(14) << std::setprecision(2) << r.gflops
            << "\n";

        csv
            << r.M << ","
            << r.N << ","
            << r.K << ","
            << r.best_ms << ","
            << r.avg_ms << ","
            << r.worst_ms << ","
            << r.gflops
            << "\n";
    }

    csv.close();

    std::cout
        << "\nBenchmark complete.\n";

    std::cout
        << "Results written to benchmark_results.csv\n";

    return 0;
}