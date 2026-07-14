#pragma once

#include <cstddef>

namespace renorm
{
namespace runtime
{

struct RegionScheduleItem
{
    std::size_t region;

    std::size_t level;

    float priority;
};

}
}