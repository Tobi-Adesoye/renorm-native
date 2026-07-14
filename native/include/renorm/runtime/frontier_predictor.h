#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// FrontierPredictor
//
// Predicts which regions will be expanded next.
//
// Phase E5
//==============================================================

class FrontierPredictor
{
public:

    void predict(
        ExecutionContext& context);

private:

    float compute_score(
        const CompiledRegion& region) const;

    void normalize(
        ExecutionContext& context);
};

}
}