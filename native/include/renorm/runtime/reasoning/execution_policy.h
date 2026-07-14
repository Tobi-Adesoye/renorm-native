#pragma once

#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

struct ExecutionPolicy
{
    std::vector<bool> activeBlocks;

    std::vector<bool> blockedBlocks;

    bool allowExpansion = true;

    bool stopExecution = false;

    float confidence = 0.0f;

    float reductionEstimate = 0.0f;
};

}
}