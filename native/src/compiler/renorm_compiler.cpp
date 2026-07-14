#include "renorm/compiler/renorm_compiler.h"

#include "passes/pass_dead_end.h"
#include "passes/pass_reachability.h"
#include "passes/pass_chain_compression.h"
#include "passes/pass_equivalence_folding.h"

namespace renorm
{
namespace compiler
{

Graph RenormCompiler::compile(
    const Graph& input)
{
    Graph output = input;

    //----------------------------------------------------------
    // Pass 1
    //----------------------------------------------------------

    PassDeadEnd deadEnd;

    output =
        deadEnd.run(
            output,
            output.goal_node());

    //----------------------------------------------------------
    // Pass 2
    //----------------------------------------------------------

    PassReachability reachability;

    output =
        reachability.run(
            output,
            output.goal_node());

    //----------------------------------------------------------
    // Pass 3
    //----------------------------------------------------------

    PassChainCompression chainCompression;

    output =
        chainCompression.run(output);

    //----------------------------------------------------------
    // Pass 4
    //----------------------------------------------------------

    PassEquivalenceFolding equivalence;

    output =
        equivalence.run(output);

    return output;
}

}
}