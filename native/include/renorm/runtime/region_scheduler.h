#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// RegionScheduler
//
// Builds an executable schedule from the Region A* path.
//
// Phase F4
//==============================================================

class RegionScheduler
{
public:

    void schedule(
        ExecutionContext& context);

private:

    void compute_levels(
        ExecutionContext& context);

    void assign_priorities(
        ExecutionContext& context);

    void build_execution_queue(
        ExecutionContext& context);
};

}
}