#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// DeadRegionPredictor
//
// Predicts regions that are unlikely to contribute to the
// remaining search.
//
// Produces:
//
//  • dead regions
//  • exhausted regions
//  • prune candidates
//
// The AdaptivePruner consumes these predictions.
//==============================================================

class DeadRegionPredictor
{
public:

    DeadRegionPredictor() = default;

    //----------------------------------------------------------
    // Execute prediction
    //----------------------------------------------------------

    void predict(
        ExecutionContext& context) const;

private:

    bool is_dead(
        const ExecutionContext& context,
        std::size_t regionId) const;

    float dead_score(
        const ExecutionContext& context,
        std::size_t regionId) const;
};

}
}