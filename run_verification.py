import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from renorm.layers import RenormLinear, RenormBlock
from renorm.containers import RenormContainer

print("=== SYSTEM CHECK: IMPORT SUCCESSFUL ===")
dim = 64
depth = 100
container = RenormContainer(depth=depth, dim=dim)
x = torch.randn(1, dim, requires_grad=True)

out = container(x)
out.mean().backward()

print(f"Container Depth: {depth}")
print(f"Calculated Initial Beta: {container.beta[0].item():.6f}")
print(f"Outbound Gradient Variance: {x.grad.var().item():.6f}")
print("=======================================")
