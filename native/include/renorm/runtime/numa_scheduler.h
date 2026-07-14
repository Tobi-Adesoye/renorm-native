#pragma once

#include <cstddef>
#include <vector>

namespace renorm
{
namespace runtime
{

//==============================================================
//
// NUMAScheduler
//
// Maps worker threads to NUMA nodes.
//
// Current:
//
//   • Portable fallback
//
// Future:
//
//   • Linux libnuma
//   • Windows GroupAffinity
//   • hwloc integration
//
//==============================================================

class NUMAScheduler
{
public:

    NUMAScheduler();

    //----------------------------------------------------------
    // Detect topology
    //----------------------------------------------------------

    void initialize(
        std::size_t workerCount);

    //----------------------------------------------------------
    // NUMA node for worker
    //----------------------------------------------------------

    std::size_t node(
        std::size_t workerId) const;

    //----------------------------------------------------------
    // Workers on node
    //----------------------------------------------------------

    const std::vector<std::size_t>&
    workers(
        std::size_t numaNode) const;

    //----------------------------------------------------------
    // Number of NUMA nodes
    //----------------------------------------------------------

    std::size_t node_count() const noexcept;

private:

    std::size_t nodeCount_ = 1;

    std::vector<std::size_t> workerNode_;

    std::vector<
        std::vector<std::size_t>>
        nodeWorkers_;
};

} // namespace runtime
} // namespace renorm