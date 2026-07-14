#pragma once

#include "renorm/thread_pool.h"
#include "renorm/task_graph.h"

namespace renorm
{

class GraphExecutor
{
public:

    explicit GraphExecutor(ThreadPool& pool)
        : pool_(pool) {}

    void run(TaskGraph& graph)
    {
        for (auto& task : graph.tasks())
        {
            try_schedule(task);
        }
    }

private:

    void try_schedule(Task& task)
    {
        if (task.deps.load() == 0)
        {
            pool_.submit([&task](WorkerContext& ctx)
            {
                task.fn(task);

                for (auto* dep : task.dependents)
                {
                    if (--dep->deps == 0)
                        ; // would re-schedule via pool (hooked externally)
                }
            });
        }
    }

private:

    ThreadPool& pool_;
};

} // namespace renorm