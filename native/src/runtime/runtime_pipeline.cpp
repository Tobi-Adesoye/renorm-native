#include "renorm/runtime/runtime_pipeline.h"

#include "renorm/runtime/execution_context.h"

namespace renorm
{
namespace runtime
{

//==============================================================

void RuntimePipeline::add_stage(
    std::unique_ptr<RuntimeStage> stage)
{
    stages_.push_back(
        std::move(stage));
}

//==============================================================

void RuntimePipeline::execute(
    ExecutionContext& context)
{
    for (auto& stage : stages_)
    {
        stage->execute(
            context);

        if (context.terminate)
            break;
    }
}

//==============================================================

void RuntimePipeline::clear()
{
    stages_.clear();
}

//==============================================================

std::size_t RuntimePipeline::size() const
{
    return stages_.size();
}

}
}