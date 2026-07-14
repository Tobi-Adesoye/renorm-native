#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
//
// PatternDetector
//
// Detects structural runtime patterns.
//
// Detected:
//
//  Corridor regions
//  Hub regions
//  Bottlenecks
//  Dense regions
//  Sparse regions
//  Repeated traversal
//  Branching characteristics
//
//==============================================================

class PatternDetector
{
public:

    PatternDetector() = default;

    //----------------------------------------------------------

    void detect(
        ExecutionContext& context) const;

private:

    void detect_corridors(
        ExecutionContext& context) const;

    void detect_hubs(
        ExecutionContext& context) const;

    void detect_bottlenecks(
        ExecutionContext& context) const;

    void detect_density(
        ExecutionContext& context) const;

    void detect_repeated_traversal(
        ExecutionContext& context) const;
};

}
}