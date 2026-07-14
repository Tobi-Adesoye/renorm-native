#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// RegionAnalyser
//
// Computes structural metrics for every compiled region.
//
// Phase E1
//==============================================================

class RegionAnalyser
{
public:

    void analyse(
        ExecutionContext& context);

private:

    void compute_density(
        ExecutionContext& context);

    void compute_connectivity(
        ExecutionContext& context);

    void compute_branching(
        ExecutionContext& context);

    void compute_boundary(
        ExecutionContext& context);
};

}
}