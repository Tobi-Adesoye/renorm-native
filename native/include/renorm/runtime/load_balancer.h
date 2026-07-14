#pragma once

#include <cstddef>

namespace renorm
{
namespace runtime
{

class AdaptiveScheduler;

//==============================================================
//
// LoadBalancer
//
// Dynamic runtime load balancing.
//
// Responsibilities:
//
// • Monitor worker queue pressure
// • Redistribute work before starvation
// • Minimize worker idle time
// • Reduce stealing frequency
//
//==============================================================

class LoadBalancer
{
public:

    explicit LoadBalancer(
        AdaptiveScheduler& scheduler);

    //----------------------------------------------------------
    // Perform one balancing pass
    //----------------------------------------------------------

    void balance();

private:

    std::size_t find_most_loaded() const;

    std::size_t find_least_loaded() const;

private:

    AdaptiveScheduler& scheduler_;
};

} // namespace runtime
} // namespace renorm