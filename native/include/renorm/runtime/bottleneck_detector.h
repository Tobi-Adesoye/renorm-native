#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// BottleneckDetector
//
// Detects articulation-like regions.
//
// Phase E4
//==============================================================

class BottleneckDetector
{
public:

    void detect(
        ExecutionContext& context);

private:

    bool is_bottleneck(
        const CompiledRegion& region,
        const ExecutionContext& context) const;
};

}
}