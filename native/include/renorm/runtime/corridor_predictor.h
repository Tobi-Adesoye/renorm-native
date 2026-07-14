#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// CorridorPredictor
//
// Predicts structural bottlenecks and future execution
// corridors.
//
// Updates:
//
//   • corridorNodes
//   • hubNodes
//   • bottleneckNodes
//   • frontierDensity
//   • frontierQuality
//   • graphComplexity
//==============================================================

class CorridorPredictor
{
public:

    CorridorPredictor() = default;

    //----------------------------------------------------------
    // Execute prediction
    //----------------------------------------------------------

    void predict(
        ExecutionContext& context) const;

private:

    float compute_density(
        const ExecutionContext& context) const;

    float compute_quality(
        const ExecutionContext& context) const;

    float compute_complexity(
        const ExecutionContext& context) const;

    bool is_corridor(
        const ExecutionContext& context,
        std::size_t regionId) const;

    bool is_hub(
        const ExecutionContext& context,
        std::size_t regionId) const;

    bool is_bottleneck(
        const ExecutionContext& context,
        std::size_t regionId) const;
};

}
}