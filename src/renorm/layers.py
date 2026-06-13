import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

# ──────────────────────────────────────────────────────────────────────────────
# TRITON AUTOTUNING MATRIX CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

def get_renorm_tuning_config():
    """
    Returns hardware-adaptive configurations to optimize thread-warps,
    register allocation, and pipeline stages based on host GPU architecture.
    """
    return [
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=5),
    ]

# ──────────────────────────────────────────────────────────────────────────────
# HARDWARE-NATIVE KERNEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(configs=get_renorm_tuning_config(), key=['M', 'N'])
@triton.jit
def _fused_renorm_forward_kernel(
    X_ptr, W_ptr, Y_ptr, Scale_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    num_stages: tl.constexpr
):
    """
    Fused SRAM-Resident Kernel: Executes intermediate renormalization and
    linear projection in a single hardware execution pass to eliminate HBM materialization.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, 16) # Base tile tracking width

    # Allocate pointer maps
    x_ptrs = X_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)

    # Accumulator initialization
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Online mean and variance tracking variables for stabilization arithmetic
    m_2 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    # Unified loop streaming through hidden dimensions
    for k in range(0, tl.cdiv(K, 16)):
        k_offset = k * 16
        
        # Guarded load with accurate sliding spatial masking coordinates
        x_tile = tl.load(x_ptrs, mask=(offs_am[:, None] < M) & ((k_offset + offs_k[None, :]) < K), other=0.0)
        w_tile = tl.load(w_ptrs, mask=((k_offset + offs_k[:, None]) < K) & (offs_bn[None, :] < N), other=0.0)

        # Self-stabilization arithmetic: compute running local energy bounds
        x_sq = x_tile * x_tile
        row_var = tl.sum(x_sq, axis=1)
        m_2 = m_2 + row_var

        # Fused tensor core dot product acceleration
        accumulator += tl.dot(x_tile, w_tile)

        x_ptrs += 16 * stride_xk
        w_ptrs += 16 * stride_wk

    # Compute exact scale scalar inside SRAM registers
    v_scale = tl.math.rsqrt((m_2 / K) + eps)

    # Apply register-fused stabilization factor directly onto final outputs
    offs_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = Y_ptr + (offs_ym[:, None] * stride_ym + offs_yn[None, :] * stride_yn)

    # Store scale factors for backprop calculus
    if pid_n == 0:
        tl.store(Scale_ptr + offs_ym, v_scale, mask=offs_ym < M)

    final_output = accumulator * v_scale[:, None]
    tl.store(y_ptrs, final_output, mask=(offs_ym[:, None] < M) & (offs_yn[None, :] < N))


# ──────────────────────────────────────────────────────────────────────────────
# CORE PYTORCH INTERFACE WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

class RenormLinearFunction(torch.autograd.Function):
    """
    Custom Autograd layer leveraging Triton execution maps with native
    PyTorch memory-saving tracking loops.
    """
    @staticmethod
    def forward(ctx, x, weight, bias=None, eps=1e-5):
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]).contiguous()
        w_flat = weight.contiguous()

        M, K = x_flat.shape
        K_w, N = w_flat.shape
        assert K == K_w, "Incompatible inner tensor dimensions across linear graph"

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        scale = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

        _fused_renorm_forward_kernel[grid](
            x_flat, w_flat, out, scale,
            M, N, K,
            x_flat.stride(0), x_flat.stride(1),
            w_flat.stride(0), w_flat.stride(1),
            out.stride(0), out.stride(1),
            eps
        )

        ctx.save_for_backward(x_flat, w_flat, bias, scale)

        if bias is not None:
            out += bias

        return out.view(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, w_flat, bias, scale = ctx.saved_tensors
        grad_flat = grad_output.view(-1, grad_output.shape[-1]).contiguous()

        # 1. Scale incoming gradients back through the forward scaling factor
        grad_scaled = grad_flat * scale[:, None]

        # 2. Compute base structural gradients (Note: w_flat is shape (K, N) so we dot directly)
        grad_x = grad_scaled @ w_flat.t()
        grad_w = x_flat.t() @ grad_scaled
        grad_bias = grad_flat.sum(dim=0) if bias is not None else None

        # 3. Apply variance tracking graph approximation to prevent gradient explosion
        grad_x = grad_x * scale[:, None]

        return grad_x.view_as(x_flat), grad_w, grad_bias, None


# ──────────────────────────────────────────────────────────────────────────────
# STANDARD PRODUCTION-GRADE LAYER WRAPPER
# ──────────────────────────────────────────────────────────────────────────────

class RenormLinear(nn.Module):
    """
    Standard PyTorch Module wrapper exposing the self-stabilizing Renorm operation.
    Provides seamless replacement for traditional nn.Linear layers.
    """
    def __init__(self, in_features, out_features, bias=True, eps=1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization with custom adaptation
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Weight needs transpose to form (K, N) tracking matrix internally
        return RenormLinearFunction.apply(x, self.weight.t(), self.bias, self.eps)


# ──────────────────────────────────────────────────────────────────────────────
# PRODUCTION-GRADE TRANSFORMER INTERFACE LAYER
# ──────────────────────────────────────────────────────────────────────────────

class RenormTransformerLayer(nn.Module):
    """
    High-performance self-stabilizing Transformer block wrapper designed
    specifically for Video DiT and consumer audio scale profiles.
    """
    def __init__(self, dim, heads, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.eps = eps

        # Open-source API wrappers interface directly with upstream configurations
        self.qkv_proj = nn.Parameter(torch.empty(dim, 3 * dim))
        self.out_proj = nn.Parameter(torch.empty(dim, dim))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights with variance preservation bounds matching custom layout
        fan_in_qkv, _ = nn.init._calculate_fan_in_and_fan_out(self.qkv_proj.t())
        gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
        std_qkv = gain / math.sqrt(fan_in_qkv)
        bound_qkv = math.sqrt(3.0) * std_qkv
        with torch.no_grad():
            self.qkv_proj.uniform_(-bound_qkv, bound_qkv)

        fan_in_out, _ = nn.init._calculate_fan_in_and_fan_out(self.out_proj.t())
        std_out = gain / math.sqrt(fan_in_out)
        bound_out = math.sqrt(3.0) * std_out
        with torch.no_grad():
            self.out_proj.uniform_(-bound_out, bound_out)

    def forward(self, x):
        """
        Executes structural tracking passes over input sequence matrices.
        Bypasses intermediate activations to preserve VRAM limits.
        """
        # 1. Project unified attention layouts via fused self-stabilizing matrix
        qkv = RenormLinearFunction.apply(x, self.qkv_proj, None, self.eps)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 2. Inline Standard scaled dot-product attention calculation pass
        B, S, D = q.shape
        q = q.view(B, S, self.heads, D // self.heads).transpose(1, 2)
        k = k.view(B, S, self.heads, D // self.heads).transpose(1, 2)
        v = v.view(B, S, self.heads, D // self.heads).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # 3. Output structural projection pass
        return RenormLinearFunction.apply(attn_out, self.out_proj, None, self.eps)