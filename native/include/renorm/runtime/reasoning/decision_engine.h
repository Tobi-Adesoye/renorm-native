#pragma once

#include <memory>

namespace renorm
{
namespace runtime
{

//--------------------------------------------------------------
// Forward declaration
//--------------------------------------------------------------

struct ExecutionContext;

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

    ReasoningEngine();
    ~ReasoningEngine();

    ReasoningEngine(const ReasoningEngine&) = delete;
    ReasoningEngine& operator=(const ReasoningEngine&) = delete;

    ReasoningEngine(ReasoningEngine&&) noexcept;
    ReasoningEngine& operator=(ReasoningEngine&&) noexcept;

    //----------------------------------------------------------
    // Execute complete reasoning pipeline
    //----------------------------------------------------------

    void execute(
        ExecutionContext& context) const;

private:

    //----------------------------------------------------------
    // Synchronize runtime metrics
    //----------------------------------------------------------

    void synchronize_metrics(
        ExecutionContext& context) const;

private:

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace runtime
} // namespace renorm