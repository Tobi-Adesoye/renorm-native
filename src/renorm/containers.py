import math

import torch
import torch.nn as nn

from .layers import RenormLinear


class RenormContainer(nn.Module):
    def __init__(self, depth: int, dim: int, hardware_alignment: int = 16):
        super().__init__()
        self.depth = depth
        self.dim = dim

        self.layers = nn.ModuleList(
            [RenormLinear(dim, dim, hardware_alignment=hardware_alignment) for _ in range(depth)]
        )

        # Calculate optimal heuristic target scale
        beta_scale = 1.0 / math.sqrt(self.dim * self.depth)

        # Convert target scale to its corresponding inverse-sigmoid proxy coordinate
        # Bound target scale slightly clear of strict 0/1 limits to prevent initial log(0) explosions
        beta_scale = max(min(beta_scale, 0.999), 0.001)
        proxy_init = math.log(beta_scale / (1.0 - beta_scale))

        self.beta_proxy = nn.Parameter(torch.ones(depth) * proxy_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass proxy values through the sigmoid manifold function natively
        functional_betas = torch.sigmoid(self.beta_proxy)

        for i, layer in enumerate(self.layers):
            x = x + (functional_betas[i] * layer(x))
        return x
