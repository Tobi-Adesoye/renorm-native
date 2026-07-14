#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// CorridorDetector
//
// Detects:
//
// • corridor regions
// • dead-end regions
// • linear chains
//
// Phase E2
//==============================================================

class CorridorDetector
{
public:

    void detect(
        ExecutionContext& context);

private:

    bool is_corridor(
        const CompiledRegion& region,
        const ExecutionContext& context) const;

    bool is_dead_end(
        const CompiledRegion& region) const;

    bool is_linear(
        const CompiledRegion& region) const;
};

}
}