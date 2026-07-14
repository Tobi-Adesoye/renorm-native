#pragma once

#include "renorm/runtime/execution_context.h"

#include <unordered_map>

namespace renorm
{
namespace runtime
{

//==============================================================
// CachedRegion
//
// Stores reusable reasoning results.
//
// Future:
//   heuristic
//   confidence
//   prediction
//   execution policy
//==============================================================

struct CachedRegion
{
    float heuristic = 0.0f;

    float confidence = 0.0f;

    float predictionConfidence = 0.0f;

    std::size_t executionPriority = 0;

    std::size_t expansionPolicy = 0;

    bool valid = false;
};

//==============================================================
// RegionCache
//==============================================================

class RegionCache
{
public:

    RegionCache() = default;

    //----------------------------------------------------------
    // Query
    //----------------------------------------------------------

    bool contains(
        std::size_t region) const;

    //----------------------------------------------------------
    // Retrieve
    //----------------------------------------------------------

    const CachedRegion* lookup(
        std::size_t region) const;

    //----------------------------------------------------------
    // Insert
    //----------------------------------------------------------

    void store(
        std::size_t region,
        const CachedRegion& data);

    //----------------------------------------------------------
    // Remove all
    //----------------------------------------------------------

    void clear();

private:

    std::unordered_map<
        std::size_t,
        CachedRegion> cache;
};

}
}