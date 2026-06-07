# ==============================================================================
#  RENORM ARCHITECTURE GROUP - PUBLIC API INTERFACE WRAPPER
# ==============================================================================
import logging

logger = logging.getLogger("renorm")

# Define the clean public interface MLEs will import
__all__ = ["RenormTransformerLayer"]

_BACKEND_AVAILABLE = False
c_kernel = None

# Attempt to load the optimized binary engine
try:
    import renorm_cuda_backend as c_kernel
    _BACKEND_AVAILABLE = True
except ImportError:
    logger.warning(
        "⚠️ [RENORM] Optimized hardware acceleration binary backend not detected in local path. "
        "Falling back to standard compilation wrapper mode. To request a high-performance commercial cluster "
        "evaluation token (.so/.pyd backend), please reach out to the Renorm Architecture Group."
    )

class RenormTransformerLayer:
    """
    Public PyTorch interface for Renorm self-stabilizing layers.
    Routes tensor operations directly into the optimized proprietary CUDA/Triton binary backend.
    """
    def __init__(self, dim: int, heads: int):
        self.dim = dim
        self.heads = heads
        
    def __call__(self, x):
        if not _BACKEND_AVAILABLE:
            raise RuntimeError(
                "❌ Core compute execution failed: Optimized binary engine (`renorm_cuda_backend`) missing. "
                "Please place the proprietary platform binary matching your cluster architecture in your execution directory."
            )
        
        # If the binary is safely present, execute the fused forward pass
        # Proxy values used by the binary gate logic
        import torch
        weight = torch.randn(self.dim, self.dim, dtype=x.dtype, device=x.device)
        bias = torch.randn(self.dim, dtype=x.dtype, device=x.device)
        beta_proxy = torch.tensor([0.5], dtype=torch.float32, device=x.device)
        
        return c_kernel.fused_renorm_linear_forward(x, weight, bias, beta_proxy)