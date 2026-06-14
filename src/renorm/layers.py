import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def get_renorm_tuning_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_warps=8, num_stages=4),
    ]

if HAS_TRITON:
    @triton.autotune(configs=get_renorm_tuning_config(), key=['M', 'N'])
    @triton.jit
    def _fused_renorm_forward_kernel_2d(
        X_ptr, W_ptr, Y_ptr, Scale_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_ym, stride_yn,
        eps,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        num_stages: tl.constexpr
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, 16)

        x_ptrs = X_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_bn[None, :] * stride_wn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        m_2 = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, 16)):
            k_offset = k * 16
            mask_x = (offs_am[:, None] < M) & ((k_offset + offs_k[None, :]) < K)
            mask_w = ((k_offset + offs_k[:, None]) < K) & (offs_bn[None, :] < N)
            
            x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
            w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

            x_tile_f32 = x_tile.to(tl.float32)
            m_2 += tl.sum(x_tile_f32 * x_tile_f32, axis=1)
            accumulator += tl.dot(x_tile, w_tile)
            
            x_ptrs += 16 * stride_xk
            w_ptrs += 16 * stride_wk

        # Triton branch: enforce numerical floor on variance
        var_val = m_2 / K
        var_floor = tl.where(var_val < eps, eps, var_val)
        v_scale = tl.math.rsqrt(var_floor + eps)

        offs_ym = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_yn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        y_ptrs = Y_ptr + (offs_ym[:, None] * stride_ym + offs_yn[None, :] * stride_yn)

        if pid_n == 0:
            tl.store(Scale_ptr + offs_ym, v_scale, mask=offs_ym < M)

        final_output = accumulator * v_scale[:, None]
        tl.store(y_ptrs, final_output, mask=(offs_ym[:, None] < M) & (offs_yn[None, :] < N))

class RenormLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, eps=1e-5):
        ctx.x_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        ctx.x_shape = x.shape

        x_flat = x.reshape(-1, x.shape[-1])
        w_flat = weight.t() 

        M, K = x_flat.shape
        K_w, N = w_flat.shape
        
        assert K == K_w, f"Dimension mismatch: Inner graph K ({K}) must match Weight K ({K_w})"

        is_power_of_two = ((K & (K - 1)) == 0) and (K > 0)
        strides_valid = x_flat.stride(0) >= 0 and x_flat.stride(1) >= 0 if x_flat.dim() == 2 else False

        if not HAS_TRITON or not torch.cuda.is_available() or x.device.type != 'cuda' or not is_power_of_two or not strides_valid:
            x_32 = x_flat.to(torch.float32)
            m_2 = torch.sum(x_32 * x_32, dim=1, dtype=torch.float32)
            
            # MATHEMATICAL ANCHOR: Force a safe variance lower boundary.
            # This completely shields rsqrt from exploding when scaling underflow matrices.
            var_val = m_2 / K
            var_floor = torch.clamp(var_val, min=eps)
            scale_32 = torch.rsqrt(var_floor + eps)
            
            w_32 = w_flat.to(torch.float32)
            out_32 = (x_32 @ w_32) * scale_32[:, None]
            
            if bias is not None:
                out_32 += bias.to(torch.float32)
            
            if x.dtype == torch.float16:
                out_32 = torch.clamp(out_32, min=-65504.0, max=65504.0)
            elif x.dtype == torch.bfloat16:
                out_32 = torch.clamp(out_32, min=-3.38e38, max=3.38e38)

            out = out_32.to(x_flat.dtype)
            ctx.save_for_backward(x_flat, weight, bias, scale_32.to(x_flat.dtype))
            return out

        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        scale = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_SIZE_M']),
            triton.cdiv(N, META['BLOCK_SIZE_N'])
        )

        _fused_renorm_forward_kernel_2d[grid](
            x_flat, w_flat, out, scale,
            M, N, K,
            x_flat.stride(0), x_flat.stride(1),
            w_flat.stride(0), w_flat.stride(1),
            out.stride(0), out.stride(1),
            eps
        )

        if bias is not None:
            out += bias

        ctx.save_for_backward(x_flat, weight, bias, scale.to(x_flat.dtype))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_flat, weight, bias, scale = ctx.saved_tensors
        
        if x_flat is None:
            return None, None, None, None

        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).to(torch.float32)
        scale_clamped = torch.clamp(scale, min=1e-7, max=1e5).to(torch.float32)
        x_flat_32 = x_flat.to(torch.float32)
        weight_32 = weight.to(torch.float32)

        grad_scaled = grad_flat * scale_clamped[:, None]
        
        grad_x = grad_scaled @ weight_32
        grad_w = grad_scaled.t() @ x_flat_32
        
        grad_bias = grad_flat.sum(dim=0) if bias is not None else None
        grad_x = grad_x * scale_clamped[:, None]

        if ctx.x_dtype == torch.float16:
            grad_x = torch.clamp(grad_x, min=-65504.0, max=65504.0)
        if ctx.weight_dtype == torch.float16:
            grad_w = torch.clamp(grad_w, min=-65504.0, max=65504.0)

        grad_x_reshaped = grad_x.view(ctx.x_shape)
        return grad_x_reshaped.to(ctx.x_dtype).clone(), grad_w.to(ctx.weight_dtype).clone(), grad_bias, None

class RenormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        orig_shape = x.shape
        res = RenormLinearFunction.apply(x, self.weight, self.bias, self.eps)
        return res.view(*orig_shape[:-1], self.out_features)