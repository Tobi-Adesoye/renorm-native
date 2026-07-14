#pragma once

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
//
// ReasoningEngine
//
// Complete runtime reasoning pipeline.
//
//      Pattern Detection
//              │
//              ▼
//      Prediction
//              │
//              ▼
//      Heuristic Evaluation
//              │
//              ▼
//      Decision Policy
//
//==============================================================

class ReasoningEngine
{
public:

    ReasoningEngine() = default;

    //----------------------------------------------------------

    void execute(
        ExecutionContext& context) const;

private:

    PatternDetector patternDetector;

    PredictionEngine predictionEngine;

    HeuristicEngine heuristicEngine;

    DecisionEngine decisionEngine;
};

}
}