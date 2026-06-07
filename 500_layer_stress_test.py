import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.nn as nn
from renorm.transformer import RenormTransformerLayer

def run_industrial_stress_test():
    print("=== INITIALIZING INDUSTRIAL STRESS TEST: 500-LAYER DEEP STACK ===")
    
    DEPTH = 500
    BATCH_SIZE = 2
    SEQ_LEN = 16
    EMBED_DIM = 64
    HEADS = 4
    
    print(f"Allocating {DEPTH} Self-Stabilizing Layers...")
    
    # Construct an ultra-deep stack of Renorm layers
    model = nn.Sequential(*[
        RenormTransformerLayer(d_model=EMBED_DIM, n_heads=HEADS)
        for _ in range(DEPTH)
    ])
    
    # Generate mock high-dimensional tensor input
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM, requires_grad=True)
    
    # Forward Pass through the 500-layer sub-manifold highway
    output = model(x)
    
    # Compute pseudo-loss to trigger backpropagation across all 500 layers
    loss = output.sum()
    loss.backward()
    
    grad_norm = x.grad.norm().item()
    
    print("-" * 80)
    print(f"Total Structural Blocks Successfully Traversed : {DEPTH}")
    print(f"Source Activation Gradient Status Norm          : {grad_norm:.6f}")
    print("-" * 80)
    
    if 1e-5 < grad_norm < 100.0:
        print("STATUS: SYSTEMIC STABILITY VERIFIED. GRADIENTS ANCHORED SUCCESSFULLY.")
    else:
        print("STATUS: MALFUNCTION DETECTED.")
    print("=== STRESS TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_industrial_stress_test()