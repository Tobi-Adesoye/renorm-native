#pragma once

//==============================================================
//
// PredictionEngine
//
// Runtime prediction stage.
//
// Responsibilities:
//
//   • Predict future frontier growth
//   • Predict expansion cost
//   • Predict remaining goal distance
//   • Predict confidence
//
// Consumes:
//
//   • PatternDetector output
//   • ExecutionContext runtime state
//
// Produces:
//
//   • Frontier prediction fields
//   • RuntimeResult prediction metrics
//
// Notes:
//
//   This engine performs prediction only.
//   Serialization of results belongs to RuntimeResultBuilder.
//
//==============================================================

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// Forward declaration
//--------------------------------------------------------------

struct ExecutionContext;

//==============================================================
// PredictionEngine
//==============================================================

class PredictionEngine
{
public:

    PredictionEngine() = default;
    ~PredictionEngine() = default;

    PredictionEngine(const PredictionEngine&) = default;
    PredictionEngine& operator=(const PredictionEngine&) = default;

    PredictionEngine(PredictionEngine&&) noexcept = default;
    PredictionEngine& operator=(PredictionEngine&&) noexcept = default;

    //----------------------------------------------------------
    // Execute complete prediction pipeline
    //----------------------------------------------------------

    void predict(
        ExecutionContext& context) const;

private:

    //----------------------------------------------------------
    // Prediction stages
    //----------------------------------------------------------

    void predict_frontier(
        ExecutionContext& context) const;

    void predict_expansion(
        ExecutionContext& context) const;

    void predict_goal(
        ExecutionContext& context) const;

    void predict_confidence(
        ExecutionContext& context) const;
};

} // namespace runtime
} // namespace renorm