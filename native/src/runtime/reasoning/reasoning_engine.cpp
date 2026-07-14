#include "renorm/runtime/reasoning/reasoning_engine.h"

#include "renorm/runtime/execution_context.h"

#include "renorm/runtime/reasoning/pattern_detector.h"
#include "renorm/runtime/reasoning/prediction_engine.h"
#include "renorm/runtime/reasoning/heuristic_engine.h"
#include "renorm/runtime/reasoning/decision_engine.h"

namespace renorm
{
namespace runtime
{

//==============================================================
// Implementation
//==============================================================

struct ReasoningEngine::Impl
{
    PatternDetector patternDetector;
    PredictionEngine predictionEngine;
    HeuristicEngine heuristicEngine;
    DecisionEngine decisionEngine;
};

//==============================================================

ReasoningEngine::ReasoningEngine()
    : impl_(std::make_unique<Impl>())
{
}

ReasoningEngine::~ReasoningEngine() = default;

ReasoningEngine::ReasoningEngine(
    ReasoningEngine&&) noexcept = default;

ReasoningEngine&
ReasoningEngine::operator=(
    ReasoningEngine&&) noexcept = default;

//==============================================================

void ReasoningEngine::execute(
    ExecutionContext& context) const
{
    //----------------------------------------------------------
    // Stage 1
    //----------------------------------------------------------

    impl_->patternDetector.detect(context);

    //----------------------------------------------------------
    // Stage 2
    //----------------------------------------------------------

    impl_->predictionEngine.predict(context);

    //----------------------------------------------------------
    // Stage 3
    //----------------------------------------------------------

    impl_->heuristicEngine.compute(context);

    //----------------------------------------------------------
    // Stage 4
    //----------------------------------------------------------

    impl_->decisionEngine.decide(context);

    //----------------------------------------------------------
    // Synchronize metrics
    //----------------------------------------------------------

    synchronize_metrics(context);
}

//==============================================================

void ReasoningEngine::synchronize_metrics(
    ExecutionContext& context) const
{
    auto& frontier = context.frontier;
    auto& result   = context.result;
    auto& policy   = context.policy;

    result.frontierDensity =
        frontier.frontierDensity;

    result.averageBranchingFactor =
        frontier.averageBranchingFactor;

    result.frontierQuality =
        frontier.frontierQuality;

    result.graphComplexity =
        frontier.graphComplexity;

    result.predictionConfidence =
        frontier.predictionConfidence;

    result.heuristicScore =
        frontier.heuristicScore;

    result.decisionCost =
        frontier.decisionCost;

    result.executionPriority =
        frontier.executionPriority;

    result.confidence =
        policy.confidence;

    result.expansionReduction =
        context.pruningRatio;

    result.expansionPolicy =
        static_cast<float>(
            static_cast<int>(
                policy.action));
}

} // namespace runtime
} // namespace renorm