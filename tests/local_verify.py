import torch
from renorm.layers import RenormLinear

# 1. Initialize our standard drop-in replacement module layer
layer = RenormLinear(in_features=128, out_features=64)

# 2. Setup testing tensors of completely different shapes (including Torsten's 2D and Transformer 3D shapes)
shapes_to_test = [
    (32, 128),     # Shape A (Torsten's shape)
    (1, 128),      # Shape B (Single inference batch item)
    (8, 24, 128)   # Shape C (Standard 3D Transformer sequential activation matrix)
]

print("\n--- Running Dynamic Shape Execution Matrix ---")
for idx, shape in enumerate(shapes_to_test):
    X = torch.randn(*shape) * 10.0
    output = layer(X)
    print(f"Test {idx + 1} | Input: {list(shape)} -> Passed! Output Shape: {list(output.shape)}")
print("=========================================================")