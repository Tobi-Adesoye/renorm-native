#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// RegionPredictor
//
// Predicts the next active execution regions.
//
// Uses:
//
//   • current frontier
//   • compiled region graph
//   • neighbouring regions
//   • execution history
//
// Updates:
//
//   frontier.predictionConfidence
//   result.predictedVisited
//==============================================================

class RegionPredictor
{
public:

    RegionPredictor() = default;

    //----------------------------------------------------------
    // Predict active regions
    //----------------------------------------------------------

    void predict(
        ExecutionContext& context) const;

private:

    float region_score(
        const ExecutionContext& context,
        std::size_t regionId) const;

    bool likely_active(
        const ExecutionContext& context,
        std::size_t regionId) const;
};

}
}