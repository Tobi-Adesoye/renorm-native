#include <cassert>
#include <cmath>
#include <iostream>

#include "renorm/gemm.h"
#include "renorm/kernel_context.h"

using namespace renorm;

int main()
{
    KernelContext ctx;

    // A = 2x3
    float A[] =
    {
        1,2,3,
        4,5,6
    };

    // B = 3x2
    float B[] =
    {
        7,8,
        9,10,
        11,12
    };

    float C[4];

    gemm(
        ctx,
        A,
        B,
        C,
        2,
        2,
        3
    );

    const float expected[] =
    {
        58,64,
        139,154
    };

    constexpr float eps = 1e-5f;

    for (int i = 0; i < 4; ++i)
    {
        assert(std::fabs(C[i] - expected[i]) < eps);
    }

    std::cout << "GEMM TEST PASSED" << std::endl;

    return 0;
}