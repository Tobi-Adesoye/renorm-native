#pragma once

namespace renorm
{

struct BlockConfig
{
    int MC;
    int NC;
    int KC;
};

// Computes hardware-adaptive block parameters using pure architectural feature fields
BlockConfig tune_for_cpu();

// Thread-safe cached reference to the calculated configuration matrix
const BlockConfig& get_block_config();

// Diagnostic logging function for performance tracking and user bug reports
void print_block_config();

} // namespace renorm