import math

import torch
import torch.nn as nn

from .attention import RenormAttention
from .layers import RenormLinear


class RenormTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * d_model

        self.attn = RenormAttention(d_model=d_model, n_heads=n_heads)
        self.mlp_gate = RenormLinear(d_model, self.dim_feedforward)
        self.mlp_down = RenormLinear(self.dim_feedforward, d_model)

        init_val = math.log(0.01 / 0.99)
        self.beta_mlp_proxy = nn.Parameter(torch.tensor([init_val]))

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        # Pass the causal flag down into the upgraded attention layer
        x = self.attn(x, causal=causal)

        mlp_res = self.mlp_down(self.mlp_gate(x))
        beta_mlp = torch.sigmoid(self.beta_mlp_proxy)

        return x + (beta_mlp * mlp_res)
