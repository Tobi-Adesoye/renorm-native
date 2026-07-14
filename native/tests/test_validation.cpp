#include "renorm/gemm.h"
#include "reference_gemm.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace renorm;

namespace
{

constexpr float ABS_TOL = 1e-5f;
constexpr float REL_TOL = 1e-4f;

bool validate_case(int M, int N, int K)
{
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);

    std::vector<float> C(M * N);
    std::vector<float> Ref(M * N);

    std::mt19937 rng(42);

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float& x : A)
        x = dist(rng);

    for (float& x : B)
        x = dist(rng);

    KernelContext ctx;

    gemm(ctx,
         A.data(),
         B.data(),
         C.data(),
         M,
         N,
         K);

    reference_gemm(
        A.data(),
        B.data(),
        Ref.data(),
        M,
        N,
        K);

    float maxAbs = 0.0f;
    float maxRel = 0.0f;
    float rmse = 0.0f;

    bool pass = true;

    for (size_t i = 0; i < C.size(); ++i)
    {
        float absErr =
            std::abs(C[i] - Ref[i]);

        maxAbs =
            std::max(maxAbs, absErr);

        rmse += absErr * absErr;

        // Relative error is meaningless when reference≈0.
        if (std::abs(Ref[i]) > ABS_TOL)
        {
            float relErr =
                absErr / std::abs(Ref[i]);

            maxRel =
                std::max(maxRel, relErr);

            if (relErr > REL_TOL)
                pass = false;
        }
        else
        {
            if (absErr > ABS_TOL)
                pass = false;
        }
    }

    rmse =
        std::sqrt(rmse / C.size());

    if (pass)
    {
        std::cout
            << std::setw(4) << M << " x "
            << std::setw(4) << N << " x "
            << std::setw(4) << K
            << "  PASS\n";
    }
    else
    {
        std::cout << "\nFAILED\n";

        std::cout
            << "Dimensions : "
            << M << " x "
            << N << " x "
            << K << '\n';

        std::cout
            << "Max Abs Error : "
            << maxAbs << '\n';

        std::cout
            << "Max Rel Error : "
            << maxRel << '\n';

        std::cout
            << "RMSE : "
            << rmse << '\n';
    }

    return pass;
}

}

int main()
{
    std::cout
        << "=========================================\n"
        << "RENORM GEMM VALIDATION SUITE\n"
        << "=========================================\n\n";

    bool ok = true;

    ok &= validate_case(1, 1, 1);
    ok &= validate_case(1, 17, 1);
    ok &= validate_case(17, 1, 17);
    ok &= validate_case(3, 5, 7);
    ok &= validate_case(7, 3, 5);
    ok &= validate_case(31, 33, 35);
    ok &= validate_case(63, 65, 67);

    if (ok)
    {
        std::cout
            << "\nALL VALIDATION TESTS PASSED\n";

        return 0;
    }

    return 1;
}