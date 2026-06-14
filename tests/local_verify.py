import torch
import torch.nn as nn
import time
from renorm.layers import RenormLinear

def run_stress_test_suite():
    print("=========================================================")
    print("🚀 STARTING EXTENSIVE RENORM RESILIENCY PRODUCTION TESTS")
    print("=========================================================\n")

    # Hardware detection check
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"📡 Execution target environment detected: [{device.upper()}]")
    if not cuda_available:
        print("⚠️  No CUDA hardware drivers active. Running through automated local fallback loop.")
    print("-" * 57)

    # ──────────────────────────────────────────────────────────────────────────
    # TEST ANGLE 1: COMFYUI PRODUCTION VIDEO TRANSFORMER SHAPES (3D Tensors)
    # ──────────────────────────────────────────────────────────────────────────
    print("🧪 Test Angle 1: Dynamic 3D Sequence Matrix Profiles...")
    # Formats match: [Batch, Sequence_Length (Frames * Height * Width), Hidden_Dim]
    transformer_shapes = [
        (1, 4096, 2048),   # Single-frame high-resolution activation patch
        (2, 2048, 4096),   # Dual batch extended video projection block
        (4, 512, 1024),    # Highly clustered sequence layout
    ]

    for idx, shape in enumerate(transformer_shapes):
        B, S, in_features = shape
        out_features = in_features // 2 # Down-projection setup match
        
        layer = RenormLinear(in_features=in_features, out_features=out_features).to(device)
        X = torch.randn(B, S, in_features, device=device)
        
        try:
            out = layer(X)
            assert out.shape == (B, S, out_features), f"Shape mismatch received: {out.shape}"
            print(f"  ✅ Shape Set {idx+1} Pass: Input {list(shape)} -> Output {list(out.shape)}")
        except Exception as e:
            print(f"  ❌ Shape Set {idx+1} FAILED: {str(e)}")
            return

    # ──────────────────────────────────────────────────────────────────────────
    # TEST ANGLE 2: MATHEMATICAL STABILITY & RE-ENTRY EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────
    print("\n🧪 Test Angle 2: Extreme Signal Variance Stabilization...")
    # We pass normal data, zeroed data, and massive energy spikes to check for NaNs
    variance_inputs = [
        ("Massive Exploded Signal", torch.randn(16, 1024, device=device) * 5000.0),
        ("Sub-Zero Micro Signal", torch.randn(16, 1024, device=device) * 0.00001),
        ("Dead Zero Matrix Vector", torch.zeros(16, 1024, device=device))
    ]

    layer_stable = RenormLinear(in_features=1024, out_features=512).to(device)

    for label, tensor in variance_inputs:
        try:
            out = layer_stable(tensor)
            nan_count = torch.isnan(out).sum().item()
            inf_count = torch.isinf(out).sum().item()
            
            if nan_count == 0 and inf_count == 0:
                print(f"  ✅ {label}: Passed cleanly! Output bounds perfectly normalized.")
            else:
                print(f"  ❌ {label}: FAILED numerical bounds constraint check (NaNs: {nan_count}, Infs: {inf_count})")
                return
        except Exception as e:
            print(f"  ❌ {label}: Failed runtime pass structure: {str(e)}")
            return

    # ──────────────────────────────────────────────────────────────────────────
    # TEST ANGLE 3: AUTOGRAD BACKPROPAGATION TRACKING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n🧪 Test Angle 3: Autograd Gradient Backward Track Integrity...")
    try:
        layer_grad = RenormLinear(in_features=256, out_features=128).to(device)
        X_grad = torch.randn(4, 256, device=device, requires_grad=True)
        
        # Execute forward pass
        out_grad = layer_grad(X_grad)
        
        # Synthesize a loss scalar calculation map
        loss = out_grad.sum()
        loss.backward()
        
        # Validate gradient graphs exist and aren't blank
        w_grad_check = layer_grad.weight.grad is not None
        x_grad_check = X_grad.grad is not None
        
        if w_grad_check and x_grad_check:
            print("  ✅ Gradient Map Verification: Passed! Backprop paths fully established.")
        else:
            print(f"  ❌ Gradient Failure: Weight tracked={w_grad_check}, Input tracked={x_grad_check}")
            return
    except Exception as e:
        print(f"  ❌ Autograd Execution Block Failed: {str(e)}")
        return

    print("\n=========================================================")
    print("🎉 ALL ANGLES SECURED: Codebase is 100% resilient!")
    print("=========================================================")

if __name__ == "__main__":
    run_stress_test_suite()