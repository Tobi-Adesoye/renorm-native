#include "renorm/runtime/numa_scheduler.h"

#include <thread>

namespace renorm
{
namespace runtime
{

//==============================================================

NUMAScheduler::NUMAScheduler() = default;

//==============================================================

void NUMAScheduler::initialize(
    std::size_t workerCount)
{
    //----------------------------------------------------------
    // Portable implementation.
    //
    // Future versions should query:
    //
    //   Linux  -> libnuma
    //   Windows-> GetNumaHighestNodeNumber
    //   hwloc
    //
    //----------------------------------------------------------

    nodeCount_ = 1;

    workerNode_.assign(
        workerCount,
        0);

    nodeWorkers_.clear();
    nodeWorkers_.resize(nodeCount_);

    for (std::size_t worker = 0;
         worker < workerCount;
         ++worker)
    {
        nodeWorkers_[0].push_back(
            worker);
    }
}

//==============================================================

std::size_t
NUMAScheduler::node(
    std::size_t workerId) const
{
    return workerNode_[workerId];
}

//==============================================================

const std::vector<std::size_t>&
NUMAScheduler::workers(
    std::size_t numaNode) const
{
    return nodeWorkers_[numaNode];
}

//==============================================================

std::size_t
NUMAScheduler::node_count() const noexcept
{
    return nodeCount_;
}

} // namespace runtime
} // namespace renorm