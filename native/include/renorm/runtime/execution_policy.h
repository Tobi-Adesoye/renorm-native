#pragma once

#include <cstddef>

namespace renorm
{
namespace runtime
{

// Forward declaration to break the compile-time include cycle with execution_context.h
struct ExecutionContext;

//==============================================================
// ExecutionAction
//==============================================================

enum class ExecutionAction
{
    Execute,
    Delay,
    Prune,
    Parallelize
};

//==============================================================
// ExecutionPolicy
//==============================================================

struct ExecutionPolicy
{
    ExecutionAction action =
        ExecutionAction::Execute;

    float confidence = 0.0f;

    float heuristic = 0.0f;

    float prediction = 0.0f;

    std::size_t priority = 0;
};

//==============================================================
// ExecutionPolicyEngine
//==============================================================

class ExecutionPolicyEngine
{
public:

    ExecutionPolicyEngine() = default;

    //----------------------------------------------------------
    // Build execution policy
    //----------------------------------------------------------

    ExecutionPolicy build(
        const ExecutionContext& context) const;

private:

    ExecutionAction choose_action(
        const ExecutionContext& context) const;

    float compute_confidence(
        const ExecutionContext& context) const;

    float compute_heuristic(
        const ExecutionContext& context) const;

    float compute_prediction(
        const ExecutionContext& context) const;

    std::size_t compute_priority(
        const ExecutionContext& context) const;
};

} // namespace runtime
} // namespace renorm