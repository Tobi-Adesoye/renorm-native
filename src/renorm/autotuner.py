import triton
import triton.language as tl

def get_renorm_tuning_config():
    """
    Returns a suite of hardware-agnostic execution configurations.
    Optimizes for different thread blocks to maximize register utilization 
    and prevent L1/SRAM cache thrashing.
    """
    return [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'num_warps': 4, 'num_stages': 2}, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'num_warps': 4, 'num_stages': 3}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'num_warps': 8, 'num_stages': 3}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'num_warps': 8, 'num_stages': 4}, num_stages=4),
        # High-leverage configuration for H100/A100 Tensor Cores
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'num_warps': 8, 'num_stages': 5}, num_stages=5),
    ]

def renorm_autotune():
    """
    Autotuning decorator to wrap your custom hardware-native Triton kernels.
    """
    return triton.autotune(
        configs=get_renorm_tuning_config(),
        key=['M', 'N'],
    )