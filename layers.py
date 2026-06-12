"""
Renorm-Native: Self-Stabilizing Layers & Triton Bindings

This module implements the mathematical core of the network. It executes the
clamped RMS self-stabilization calculations inside highly parallelized, auto-tuned GPU matrices.
"""

import os
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, Optional, Any

# =====================================================================
# TRITON GRAPH IMPLEMENTATION & COMPRESSOR OVERRIDES
# =====================================================================
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    # -----------------------------------------------------------------
    # Optimization 3: Machine-Aware Auto-Tuning (Forward Pass)
    # -----------------------------------------------------------------
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'num_warps': 8}, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_warps': 4}, num_stages=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_warps': 2}, num_stages=2),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def fused_renorm_linear_forward_kernel(
        X_ptr, W_ptr, B_ptr, OUT_ptr,
        stride_xm, stride_xk, stride_wk, stride_wn, stride_outm, stride_outn,
        M, N, K, beta,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """
        Optimization 1: Single-Pass Fused Forward Kernel
        Computes the running sum of squares and matrix multiplication in a single loop,
        saving 50% of the HBM memory load requests.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N

        # Accumulators in registers
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        var_accum = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Single-pass loop: load tiles of X and W only once
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k_idx * BLOCK_K + offs_k
            mask_k = k_offs < K
            
            x_val = tl.load(X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w_val = tl.load(W_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            
            # Accumulate sum of squares and unscaled matrix product concurrently
            var_accum += tl.sum(x_val * x_val, axis=1)
            accumulator += tl.dot(x_val, w_val)

        # Calculate final stabilization math in registers
        rms = tl.sqrt(var_accum / K)
        stabilizer = tl.maximum(rms, beta)

        # Post-scale matrix products directly using the calculated stabilizer
        accumulator = accumulator / stabilizer[:, None]

        # Add bias vector
        if B_ptr is not None:
            bias_vals = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
            accumulator += bias_vals[None, :]

        # Stream final stabilized result back to global memory exactly once
        out_offset = offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
        tl.store(OUT_ptr + out_offset, accumulator, mask=mask_m[:, None] & mask_n[None, :])


    # -----------------------------------------------------------------
    # Optimization 2: Fused Element-Wise Backward dX Kernel
    # -----------------------------------------------------------------
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'num_warps': 4}, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32, 'num_warps': 2}, num_stages=4),
        ],
        key=['M', 'K'],
    )
    @triton.jit
    def fused_renorm_backward_dx_kernel(
        X_ptr, DS_ptr, DX_ptr,
        stride_xm, stride_xk, stride_dsm, stride_dsk, stride_dxm, stride_dxk,
        M, K, beta,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """
        Fuses the analytical derivative of the self-stabilizing layers.
        Prevents intermediate memory allocations from being written to HBM during backpropagation.
        """
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M

        # Step 1: Compute RMS(X) for each row to find the stabilizer
        var_accum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k_idx * BLOCK_K + offs_k
            mask_k = k_offs < K
            x_val = tl.load(X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            var_accum += tl.sum(x_val * x_val, axis=1)

        rms = tl.sqrt(var_accum / K)
        stabilizer = tl.maximum(rms, beta)

        # Step 2: Compute the row-wise reduction: sum(dRenorm * X)
        dot_accum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k_idx * BLOCK_K + offs_k
            mask_k = k_offs < K
            x_val = tl.load(X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            ds_val = tl.load(DS_ptr + offs_m[:, None] * stride_dsm + k_offs[None, :] * stride_dsk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            dot_accum += tl.sum(ds_val * x_val, axis=1)

        # Step 3: Compute final dX tile-by-tile and stream back to HBM
        mask_scale = tl.where(rms >= beta, 1.0, 0.0)
        
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            k_offs = k_idx * BLOCK_K + offs_k
            mask_k = k_offs < K
            
            x_val = tl.load(X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            ds_val = tl.load(DS_ptr + offs_m[:, None] * stride_dsm + k_offs[None, :] * stride_dsk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            
            # Analytical derivative of RMS self-stabilization:
            # dX = (dRenorm / stabilizer) - (X * sum(dRenorm * X) * mask) / (K * stabilizer^3)
            dx_val = (ds_val / stabilizer[:, None]) - (
                x_val * dot_accum[:, None] * mask_scale[:, None]
            ) / (K * stabilizer[:, None] * stabilizer[:, None] * stabilizer[:, None])
            
            tl.store(DX_ptr + offs_m[:, None] * stride_dxm + k_offs[None, :] * stride_dxk, dx_val, mask=mask_m[:, None] & mask_k[None, :])


class FusedRenormLinearFunction(Function):
    """
    Coordinates parallel Triton registers with Python autograd mathematical operations.
    Exposes high-performance tensor-core multiplication alongside custom element-wise fusion.
    """
    @staticmethod
    def forward(ctx, X: torch.Tensor, W: torch.Tensor, B: Optional[torch.Tensor], beta: float) -> torch.Tensor:
        ctx.save_for_backward(X, W, B)
        ctx.beta = beta

        # Flatten input dims
        X_2d = X.view(-1, X.shape[-1])
        M, K = X_2d.shape
        K_w, N = W.shape
        assert K == K_w, "Dimension mismatch."

        out = torch.empty((M, N), device=X.device, dtype=X.dtype)

        if HAS_TRITON and X.is_cuda:
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
            # High-performance PyTorch compiled fallback path
            rms = torch.sqrt(torch.mean(X_2d ** 2, dim=-1, keepdim=True))
            stabilizer = torch.clamp(rms, min=beta)
            stabilized_x = X_2d / stabilizer
            
            ctx.stabilizer = stabilizer
            ctx.stabilized_x = stabilized_x
            
            out = torch.addmm(B, stabilized_x, W) if B is not None else torch.mm(stabilized_x, W)

        return out.view(*X.shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None]:
        X, W, B = ctx.saved_tensors
        beta = ctx.beta
        
        grad_flat = grad_output.view(-1, grad_output.shape[-1])
        X_flat = X.view(-1, X.shape[-1])
        
        # Calculate stabilized values
        rms = torch.sqrt(torch.mean(X_flat ** 2, dim=-1, keepdim=True))
        stabilizer = torch.clamp(rms, min=beta)
        stabilized_x = X_flat / stabilizer

        # 1. Compute weight and bias gradients
        grad_W = torch.mm(stabilized_x.t(), grad_flat)
        grad_B = grad_flat.sum(dim=0) if B is not None else None

        # 2. Compute intermediate input projection gradient (using tensor-core accelerated GEMM)
        grad_stabilized = torch.mm(grad_flat, W.t())

        # 3. Compute final dX
        grad_X = torch.empty_like(X_flat)
        
        if HAS_TRITON and X.is_cuda:
            grid = lambda meta: (
                triton.cdiv(X_flat.shape[0], meta["BLOCK_M"]),
            )
            fused_renorm_backward_dx_kernel[grid](
                X_flat, grad_stabilized, grad_X,
                X_flat.stride(0), X_flat.stride(1),
                grad_stabilized.stride(0), grad_stabilized.stride(1),
                grad_X.stride(0), grad_X.stride(1),
                X_flat.shape[0], X_flat.shape[1], beta
            )
        else:
            # CPU/Non-CUDA PyTorch fallback
            mask = (rms >= beta).float()
            inner_deriv = (grad_stabilized / stabilizer) - (
                X_flat * (grad_stabilized * X_flat).sum(dim=-1, keepdim=True) * mask
            ) / (X_flat.shape[-1] * (stabilizer ** 3))
            grad_X = inner_deriv

        return grad_X.view_as(X), grad_W, grad_B, None


class RenormSelfStabilizingLayer(nn.Module):
    """
    Standard Module API exposing our self-stabilizing mathematics to PyTorch workflows.
    """
    def __init__(self, in_features: int, out_features: int, beta: float = 0.05, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta = beta
        
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FusedRenormLinearFunction.apply(x, self.weight, self.bias, self.beta)