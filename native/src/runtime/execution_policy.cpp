#include "renorm/runtime/execution_policy.h"

namespace renorm
{
namespace runtime
{

ExecutionPolicy
ExecutionPolicyEngine::build(
    const ExecutionContext& context) const
{
    ExecutionPolicy policy;

    policy.action =
        choose_action(context);

    policy.confidence =
        compute_confidence(context);

    policy.heuristic =
        compute_heuristic(context);

    policy.prediction =
        compute_prediction(context);

    policy.priority =
        compute_priority(context);

    return policy;
}

//==============================================================

ExecutionAction
ExecutionPolicyEngine::choose_action(
    const ExecutionContext& context) const
{
    (void)context;

    //----------------------------------------------------------
    // Phase D compatibility
    //----------------------------------------------------------

    return ExecutionAction::Execute;
}

//==============================================================

float
ExecutionPolicyEngine::compute_confidence(
    const ExecutionContext& context) const
{
    (void)context;

    return 0.0f;
}

//==============================================================

float
ExecutionPolicyEngine::compute_heuristic(
    const ExecutionContext& context) const
{
    (void)context;

    return 0.0f;
}

//==============================================================

float
ExecutionPolicyEngine::compute_prediction(
    const ExecutionContext& context) const
{
    (void)context;

    return 0.0f;
}

//==============================================================

std::size_t
ExecutionPolicyEngine::compute_priority(
    const ExecutionContext& context) const
{
    (void)context;

    return 0;
}

}
}