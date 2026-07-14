#pragma once

#include <memory>
#include <vector>

namespace renorm
{
namespace runtime
{

class ExecutionContext;

//==============================================================
//
// RuntimeStage
//
// Abstract runtime stage.
//
//==============================================================

class RuntimeStage
{
public:

    virtual ~RuntimeStage() = default;

    virtual void execute(
        ExecutionContext& context) = 0;
};

//==============================================================
//
// RuntimePipeline
//
// Ordered execution pipeline.
//
//==============================================================

class RuntimePipeline
{
public:

    RuntimePipeline() = default;

    //----------------------------------------------------------

    void add_stage(
        std::unique_ptr<RuntimeStage> stage);

    //----------------------------------------------------------

    void execute(
        ExecutionContext& context);

    //----------------------------------------------------------

    void clear();

    //----------------------------------------------------------

    std::size_t size() const;

private:

    std::vector<
        std::unique_ptr<RuntimeStage>>
        stages_;
};

}
}