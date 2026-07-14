#pragma once

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

class RegionPredictor;
class ExpansionPredictor;
class DeadRegionPredictor;
class CorridorPredictor;
class GoalPredictor;

//==============================================================
// PredictionRuntime
//
// Executes the complete runtime prediction pipeline.
//
// ExecutionContext
//        │
//        ▼
// PredictionRuntime
//        │
//        ├── RegionPredictor
//        ├── ExpansionPredictor
//        ├── DeadRegionPredictor
//        ├── CorridorPredictor
//        └── GoalPredictor
//==============================================================

class PredictionRuntime
{
public:

    PredictionRuntime() = default;

    //----------------------------------------------------------
    // Execute prediction pipeline
    //----------------------------------------------------------

    void execute(
        ExecutionContext& context) const;

private:

    void normalize_predictions(
        ExecutionContext& context) const;
};

}
}