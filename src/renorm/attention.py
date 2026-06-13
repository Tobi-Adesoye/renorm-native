import torch
import torch.nn as nn
import math
from .layers import RenormLinear

class RenormAttention(nn.Module):
    """
    Multi-Head Attention Layer integrated into a Renorm-Native functional manifold
    with native Causal Mask support for autoregressive generation.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be perfectly divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = RenormLinear(d_model, d_model)
        
        init_val = math.log(0.01 / 0.99)
        self.beta_proxy = nn.Parameter(torch.tensor([init_val]))

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal masking if requested (prevents looking into the future)
        if causal:
            # Create a matrix with -inf above the diagonal
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            scores = scores + mask  # Broadcasts across Batch and Heads cleanly
            
        attn_weights = torch.functional.F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        
        transformed_context = self.out_proj(context)
        beta = torch.sigmoid(self.beta_proxy)
        
        return x + (beta * transformed_context)