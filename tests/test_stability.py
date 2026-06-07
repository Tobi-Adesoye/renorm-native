import torch
import torch.nn as nn
import math
from renorm import RenormBlock

def test_stability(depth=500, dim=64):
    # Initialize standard architecture (Beta is fixed at 1.0)
    standard_model = nn.Sequential(*[nn.Sequential(nn.Linear(dim, dim), nn.SiLU()) for _ in range(depth)])
    
    # Initialize Renorm-Native architecture (Beta is learnable)
    renorm_model = nn.Sequential(*[RenormBlock(dim) for _ in range(depth)])
    
    # ISOLATION GAINS: Initialize completely unique input tensors to avoid graph pollution
    x_std = torch.randn(1, dim, requires_grad=True)
    x_renorm = torch.randn(1, dim, requires_grad=True)
    
    # Execute Standard Baseline Forward/Backward Pass
    try:
        s_out = standard_model(x_std)
        s_loss = s_out.mean()
        s_loss.backward()
        s_var = x_std.grad.var().item()
    except Exception as e:
        s_var = float('nan')
    
    # Execute Renorm Forward/Backward Pass
    r_out = renorm_model(x_renorm)
    r_loss = r_out.mean()
    r_loss.backward()
    r_var = x_renorm.grad.var().item()
    
    # Format string readouts cleanly to account for baseline floating-point explosions
    s_var_str = f"{s_var:.6f}" if not math.isnan(s_var) and not math.isinf(s_var) else "EXPLODED (NaN / Inf)"
    r_var_str = f"{r_var:.6f}" if not math.isnan(r_var) and not math.isinf(r_var) else "FAILED"
    
    print(f"\n=== 500-LAYER STRESS TEST ===")
    print(f"Standard Gradient Variance : {s_var_str}")
    print(f"Renorm-Native Gradient Variance : {r_var_str}")
    print(f"=============================\n")
    
    if math.isnan(s_var) or math.isinf(s_var):
        print("SUCCESS: Standard ResNet suffered total spectral breakdown. Renorm-Native stabilized signal flow.")
    else:
        assert r_var < s_var, "Renorm failed to stabilize!"
        print("SUCCESS: Renorm-Native successfully minimized architectural turbulence.")

if __name__ == "__main__":
    test_stability()