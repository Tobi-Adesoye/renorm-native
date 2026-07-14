#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// AdaptivePruner
//
// Removes low-value regions from the execution schedule.
//
// ExecutionContext
//        │
//        ▼
// AdaptivePruner
//        │
//        ▼
// Filtered Schedule
//==============================================================

class AdaptivePruner
{
public:

    AdaptivePruner() = default;

    //----------------------------------------------------------
    // Adaptive pruning
    //----------------------------------------------------------

    void update(
        ExecutionContext& context) const;

private:

    //----------------------------------------------------------
    // Region scoring
    //----------------------------------------------------------

    bool should_prune(
        const ExecutionContext& context,
        std::size_t regionId) const;

    float score(
        const ExecutionContext& context,
        std::size_t regionId) const;

    float threshold(
        const ExecutionContext& context) const;
};

}
}