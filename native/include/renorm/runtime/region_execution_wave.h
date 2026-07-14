#pragma once

#include <vector>
#include <cstddef>

namespace renorm
{
namespace runtime
{

//==============================================================
// RegionExecutionWave
//
// Regions inside one wave may execute concurrently.
//==============================================================

struct RegionExecutionWave
{
    std::size_t level = 0;

    std::vector<std::size_t> regions;
};

}
}