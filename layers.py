import torch
import torch.nn as nn
import math

class RenormLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, hardware_alignment: int = 16, bias: bool = True):
        super().__init__()
        self.in_features = self._align_dim(in_features, hardware_alignment)
        self.out_features = self._align_dim(out_features, hardware_alignment)
        
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter('bias', None)
            
        self.activation = nn.SiLU()
        self._reset_parameters()

    def _align_dim(self, dim: int, alignment: int) -> int:
        return int(math.ceil(dim / alignment) * alignment)

    def _reset_parameters(self):
        nn.init.orthogonal_(self.weight, gain=1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(torch.nn.functional.linear(x, self.weight, self.bias))

class RenormBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fn = RenormLinear(dim, dim)
        
        # We initialize the proxy weight so that sigmoid(beta_proxy) is exactly 0.01
        # log(0.01 / (1 - 0.01)) = log(0.01 / 0.99)
        init_val = math.log(0.01 / 0.99)
        self.beta_proxy = nn.Parameter(torch.tensor([init_val]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce the functional manifold: mathematically bounded between (0.0, 1.0)
        beta = torch.sigmoid(self.beta_proxy)
        return x + (beta * self.fn(x))