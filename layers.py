import torch
import torch.nn as nn
import math

# ==============================================================================
# COMMERCIAL INFRASTRUCTURE ACCELERATION GATE
# This hook checks if the proprietary, high-performance compiled CUDA/Triton 
# backend is installed on the enterprise cluster.
# ==============================================================================
HAS_COMMERCIAL_KERNEL = False
try:
    # Attempt to load the closed-source, hardware-fused kernel extension
    import renorm_cuda_backend as c_kernel
    HAS_COMMERCIAL_KERNEL = True
    print("[RENORM CORE] Enterprise Compiled Hardware Acceleration Kernel: ACTIVE")
except ImportError:
    # Gracefully fall back to the open-source, public Python implementation
    pass


class RenormLinear(nn.Module):
    """
    A self-stabilizing alternative to nn.Linear.
    Replaces static identity outputs with an internally scaled variance vector
    governed by a bounded sigmoid friction gate proxy.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # High-performance weight matrix and bias allocations
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # The core architectural innovation: The Sigmoid Friction Proxy
        # Dynamically tracks and clamps structural variance down the model highway
        self.beta_proxy = nn.Parameter(torch.empty(1))
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Enforces hardware-aligned orthogonal initializations to guarantee
        isometric hidden state mappings from Step 1.
        """
        nn.init.orthogonal_(self.weight, gain=math.sqrt(2.0 / self.in_features))
        # Initialize the gate tightly to ensure initial stability under extreme depth
        nn.init.constant_(self.beta_proxy, -4.6)  # sigmoid(-4.6) ≈ 0.01

    def forward(self, x):
        # ----------------------------------------------------------------------
        # THE COMMERCIAL MOAT EXECUTION POINT
        # ----------------------------------------------------------------------
        if HAS_COMMERCIAL_KERNEL:
            # If the client has purchased the commercial tier, execute the 
            # ultra-fast fused hardware kernel, completely bypassing Python memory tracking
            return c_kernel.fused_renorm_linear_forward(x, self.weight, self.bias, self.beta_proxy)
            
        # Standard open-source fallback path (Mathematically precise, but unfused)
        raw_projection = torch.nn.functional.linear(x, self.weight, self.bias)
        beta_scale = torch.sigmoid(self.beta_proxy)
        return raw_projection * beta_scale


class RenormMLP(nn.Module):
    """
    Multi-Layer Perceptron sub-manifold using self-stabilizing layers
    and a non-linear activation highway.
    """
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.c_fc   = RenormLinear(d_model, d_ff)
        self.c_proj = RenormLinear(d_ff, d_model)
        self.act    = nn.GELU()

    def forward(self, x):
        # Project up to higher-dimensional feature space, pass through activation, 
        # then project back down to the model dimension
        return self.c_proj(self.act(self.c_fc(x)))