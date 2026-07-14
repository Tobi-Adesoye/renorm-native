#pragma once

#include "renorm/runtime/runtime_result.h"

#include <vector>
#include <string>

namespace renorm
{
namespace reasoning
{

struct ReasoningStep
{
    std::string description;

    double confidence = 0.0;

    double cost = 0.0;

    double utility = 0.0;
};

class ReasoningContext
{
public:

    //----------------------------------------------------------
    // Runtime result from Renorm
    //----------------------------------------------------------

    runtime::RuntimeResult runtime;

    //----------------------------------------------------------
    // Reasoning state
    //----------------------------------------------------------

    std::vector<ReasoningStep> steps;

    //----------------------------------------------------------
    // Final decision
    //----------------------------------------------------------

    std::string conclusion;

    double confidence = 0.0;

    //----------------------------------------------------------
    // Statistics
    //----------------------------------------------------------

    std::size_t iterations = 0;

    std::size_t hypothesesGenerated = 0;

    std::size_t hypothesesRejected = 0;

    //----------------------------------------------------------
    // Reset
    //----------------------------------------------------------

    void clear();
};

}
}