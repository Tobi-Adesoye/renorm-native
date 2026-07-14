#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// HubDetector
//
// Detects high-connectivity regions.
//
// Phase E3
//==============================================================

class HubDetector
{
public:

    void detect(
        ExecutionContext& context);

private:

    float compute_average_degree(
        const ExecutionContext& context) const;
};

}
}