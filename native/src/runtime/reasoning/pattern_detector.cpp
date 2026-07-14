#include "renorm/runtime/reasoning/pattern_detector.h"

#include <algorithm>

namespace renorm
{
namespace runtime
{

//==============================================================

void PatternDetector::detect(
    ExecutionContext& context) const
{
    detect_corridors(
        context);

    detect_hubs(
        context);

    detect_bottlenecks(
        context);

    detect_density(
        context);

    detect_repeated_traversal(
        context);
}

//==============================================================

void PatternDetector::detect_corridors(
    ExecutionContext& context) const
{
    std::size_t corridors = 0;

    for (const auto& region :
         context.graph->compiledRegions)
    {
        if (region.neighbouringRegions.size() == 2)
            ++corridors;
    }

    context.frontier.corridorNodes =
        corridors;

    context.result.corridorNodes =
        corridors;
}

//==============================================================

void PatternDetector::detect_hubs(
    ExecutionContext& context) const
{
    std::size_t hubs = 0;

    for (const auto& region :
         context.graph->compiledRegions)
    {
        if (region.neighbouringRegions.size() >= 4)
            ++hubs;
    }

    context.frontier.hubNodes =
        hubs;

    context.result.hubNodes =
        hubs;
}

//==============================================================

void PatternDetector::detect_bottlenecks(
    ExecutionContext& context) const
{
    std::size_t bottlenecks = 0;

    for (const auto& region :
         context.graph->compiledRegions)
    {
        if (region.neighbouringRegions.size() == 1)
            ++bottlenecks;
    }

    context.frontier.bottleneckNodes =
        bottlenecks;

    context.result.bottleneckNodes =
        bottlenecks;
}

//==============================================================

void PatternDetector::detect_density(
    ExecutionContext& context) const
{
    if (context.graph->compiledRegions.empty())
    {
        context.frontier.frontierDensity = 0.0f;
        context.result.frontierDensity = 0.0f;
        return;
    }

    float density = 0.0f;

    for (const auto& region :
         context.graph->compiledRegions)
    {
        density += region.density;
    }

    density /= static_cast<float>(
        context.graph->compiledRegions.size());

    context.frontier.frontierDensity =
        density;

    context.result.frontierDensity =
        density;

    //----------------------------------------------------------

    float complexity = 0.0f;

    for (const auto& region :
         context.graph->compiledRegions)
    {
        complexity +=
            static_cast<float>(
                region.neighbouringRegions.size());
    }

    complexity /= static_cast<float>(
        context.graph->compiledRegions.size());

    context.frontier.graphComplexity =
        complexity;

    context.result.graphComplexity =
        complexity;
}

//==============================================================

void PatternDetector::detect_repeated_traversal(
    ExecutionContext& context) const
{
    std::size_t repeated = 0;

    for (auto v :
         context.frontier.visited)
    {
        if (v > 1)
            ++repeated;
    }

    context.frontier.repeatedTraversalCount =
        repeated;

    context.result.repeatedTraversalCount =
        repeated;

    //----------------------------------------------------------

    float quality = 1.0f;

    if (context.frontier.nodesVisited > 0)
    {
        quality -=
            static_cast<float>(repeated) /
            static_cast<float>(
                context.frontier.nodesVisited);
    }

    quality =
        std::clamp(
            quality,
            0.0f,
            1.0f);

    context.frontier.frontierQuality =
        quality;

    context.result.frontierQuality =
        quality;
}

}
}