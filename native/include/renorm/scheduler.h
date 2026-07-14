#pragma once

#include <functional>

#include "renorm/worker_context.h"

namespace renorm
{

//--------------------------------------------------------------
// Parallel Scheduler
//--------------------------------------------------------------

void parallel_for(
    int begin,
    int end,
    int grainSize,
    const std::function<void(
        WorkerContext&,
        int,
        int)>& fn);

} // namespace renorm