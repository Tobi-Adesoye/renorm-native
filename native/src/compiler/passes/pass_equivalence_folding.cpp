#include "pass_equivalence_folding.h"

namespace renorm
{
namespace compiler
{

Graph PassEquivalenceFolding::run(
    const Graph& input) const
{
    //----------------------------------------------------------
    // Placeholder.
    //
    // Future version:
    //
    // Detect structurally equivalent subgraphs
    // and merge them into one canonical region.
    //----------------------------------------------------------

    return input;
}

}
}