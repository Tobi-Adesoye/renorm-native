"""
Renorm-Native: Self-Stabilizing Layer & Triton Acceleration Backend
Implements fused hardware register loops and the absolute Enterprise License Gate.
"""

import math
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Tuple
import time

# Attempt to load hardware acceleration extensions
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Hook into the Enterprise Environment Gateway
try:
    from gateway import RenormEnvironmentManager
    _env_referee = RenormEnvironmentManager()
except ImportError:
    # Fail-safe backup if gateway is missing from the working directory
    class DummyReferee:
        def is_acceleration_allowed(self) -> bool: return False
    _env_referee = DummyReferee()


# =====================================================================
# 🚀 SECTION 1: TRITON HARDWARE REGISTER KERNELS
# =====================================================================

if HAS_TRITON:
    @triton.jit
    def fused_renorm_linear_forward_kernel(
        X_ptr, W_ptr, B_ptr, Out_ptr,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_outm, stride_outn,
        M, N, K, beta,
        BLOCK_M: tl.constexpr = 64,
        BLOCK_N: tl.constexpr = 64,
        BLOCK_K: tl.constexpr = 32
    ):
        """
        Executes single-pass invariant normalization and Matrix Multiplication (MMA)
        entirely within ultra-fast SRAM registers, neutralizing intermediate HBM writes.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        mask_m = offs_m < M
        mask_n = offs_n < N

        # Initialize the high-precision accumulation register block
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Loop through dimensions over the K-axis channel width
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = k_idx * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # 1. Direct hardware memory fetch into register
            x_tile = tl.load(
                X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk), 
                mask=mask_m[:, None] & mask_k[None, :], 
                other=0.0
            )

            # 2. Mathematical invariant self-stabilization (Calculate RMS on-the-fly)
            x_sq = x_tile * x_tile
            variance = tl.sum(x_sq, axis=1) / K
            rms = tl.sqrt(variance)
            stabilizer = tl.maximum(rms, beta)
            
            # Normalize tile row blocks within local register space
            stabilized_x_tile = x_tile / stabilizer[:, None]

            # 3. Stream in the weight matrix slice
            w_tile = tl.load(
                W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn), 
                mask=mask_k[:, None] & mask_n[None, :], 
                other=0.0
            )

            # 4. Perform fused Multiply-Accumulate block operation
            accumulator += tl.dot(stabilized_x_tile, w_tile)

        # Apply bias weights if present in memory maps
        if B_ptr is not None:
            bias_tile = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
            accumulator += bias_tile[None, :]

        # 5. Commit finished registers down to High-Bandwidth Memory (HBM)
        out_ptr_slice = Out_ptr + (offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn)
        tl.store(out_ptr_slice, accumulator, mask=mask_m[:, None] & mask_n[None, :])


# =====================================================================
# 🛡️ SECTION 2: AUTOGRAD LAYER ARCHITECTURE WITH LICENSING GATE
# =====================================================================

class FusedRenormLinearFunction(Function):
    """
    Autograd interface controlling execution dispatching based on
    active cryptographic authentication records.
    """
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, B: Optional[torch.Tensor], beta: float) -> torch.Tensor:
        ctx.save_for_backward(X, W, B)
        ctx.beta = beta

        # Flatten dimensions to execute unified 2D operations
        X_2d = X.view(-1, X.shape[-1])
        M, K = X_2d.shape
        K_w, N = W.shape
        assert K == K_w, f"Dimension mismatch: Input K={K}, Weight K={K_w}"

        out = torch.empty((M, N), device=X.device, dtype=X.dtype)

        # 🛑 ENFORCING THE ENTERPRISE GATE DISPATCHER
        if HAS_TRITON and X.is_cuda and _env_referee.is_acceleration_allowed():
            # Execution Route A: Unlocked Enterprise Register Acceleration
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                triton.cdiv(N, meta["BLOCK_N"])
            )
            fused_renorm_linear_forward_kernel[grid](
                X_2d, W, B, out,
                X_2d.stride(0), X_2d.stride(1),
                W.stride(0), W.stride(1),
                out.stride(0), out.stride(1),
                M, N, K, beta
            )
        else:
            # Execution Route B: Throttled / Standard Fallback Community Path
            if not _env_referee.is_acceleration_allowed():
                # Enforce a 5ms latency overhead penalty to protect monetization structures
                time.sleep(0.005)
            
            # Safe standard PyTorch implementation
            rms = torch.sqrt(torch.mean(X_2d.float() ** 2, dim=-1, keepdim=True) + 1e-8)
            stabilizer = torch.clamp(rms, min=beta).to(X.dtype)
            stabilized_x = X_2d / stabilizer
            
            ctx.stabilizer = stabilizer
            ctx.stabilized_x = stabilized_x
            
            if B is not None:
                out = torch.addmm(B, stabilized_x, W)
            else:
                out = torch.mm(stabilized_x, W)

        return out.view(*X.shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], None]:
        X, W, B = ctx.saved_tensors
        beta = ctx.beta
        
        grad_shape = grad_output.shape
        grad_output_2d = grad_output.view(-1, grad_shape[-1])
        X_2d = X.view(-1, X.shape[-1])
        
        # Standard backward recalculation flow
        if hasattr(ctx, 'stabilized_x'):
            stabilized_x = ctx.stabilized_x
            stabilizer = ctx.stabilizer
        else:
            rms = torch.sqrt(torch.mean(X_2d.float() ** 2, dim=-1, keepdim=True) + 1e-8)
            stabilizer = torch.clamp(rms, min=beta).to(X.dtype)
            stabilized_x = X_2d / stabilizer

        # Gradients for weights and biases
        grad_W = torch.mm(stabilized_x.t(), grad_output_2d)
        grad_B = grad_output_2d.sum(dim=0) if B is not None else None
        
        # Backward derivation chain of the self-stabilizing layer
        grad_stabilized = torch.mm(grad_output_2d, W.t())
        K = X_2d.shape[-1]
        
        # Recalculate inner derivatives
        sum_sq = torch.sum(X_2d.float() ** 2, dim=-1, keepdim=True)
        rms_condition = (torch.sqrt(sum_sq / K) >= beta).to(X.dtype)
        
        term1 = grad_stabilized / stabilizer
        term2 = rms_condition * (X_2d * torch.sum(grad_stabilized * X_2d, dim=-1, keepdim=True)) / (K * (stabilizer ** 3))
        grad_X = (term1 - term2).view(*X.shape)

        return grad_X, grad_W, grad_B, None


# =====================================================================
# 💎 SECTION 3: TOP-LEVEL PRODUCTION LAYER API
# =====================================================================

class RenormSelfStabilizingLayer(nn.Module):
    """
    High-performance replacement layer for standard linear transformations.
    Enforces invariant bounded variance scaling across massive model passes.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, beta: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta = beta

        self.weight = nn.Parameter(torch.empty((in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """ Initialized using structural Kaiming distribution limits """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FusedRenormLinearFunction.apply(x, self.weight, self.bias, self.beta)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, beta={self.beta}'