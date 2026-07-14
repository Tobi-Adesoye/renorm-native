#pragma once

#include <atomic>
#include <functional>
#include <vector>

namespace renorm
{

struct Task;

using TaskFn = std::function<void(Task&)>;

struct Task
{
    TaskFn fn;

    std::atomic<int> deps{0};

    std::vector<Task*> dependents;
};

class TaskGraph
{
public:

    Task& create(TaskFn fn)
    {
        tasks_.emplace_back();
        tasks_.back().fn = std::move(fn);
        return tasks_.back();
    }

    void add_dependency(Task& task, Task& depends_on)
    {
        task.deps.fetch_add(1);
        depends_on.dependents.push_back(&task);
    }

    std::vector<Task>& tasks() { return tasks_; }

private:

    std::vector<Task> tasks_;
};

} // namespace renorm