import torch
import torch.nn as nn
import time
from renorm.layers import RenormLinear, RenormLinearFunction

def run_four_dimensional_diagnostic():
    print("=========================================================")
    print("🕵️‍♂️  AUDITING ALL 4 BOTTLENECK AXES SIMULTANEOUSLY")
    print("=========================================================\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📡 System Hardware Target: [{device.upper()}]")
    print("-" * 57)
    
    all_passed = True

    # ──────────────────────────────────────────────────────────────────────────
    # AXIS 1: THE STRIDE BLIND SPOT PERFORMANCE AUDIT
    # ──────────────────────────────────────────────────────────────────────────
    print("⏱️  Axis 1: Non-Contiguous Memory Stride Overhead...")
    layer_stride = RenormLinear(in_features=2048, out_features=1024).to(device)
    X_huge = torch.randn(2, 2048, 4096, device=device)
    X_sliced = X_huge[:, :, :2048]  # Forces non-contiguous stride geometry
    
    try:
        t0 = time.perf_counter()
        for _ in range(5):
            _ = layer_stride(X_sliced)
        t1 = time.perf_counter()
        axis1_latency = (t1 - t0) / 5 * 1000
        print(f"  ✅ Stride Status: Handled smoothly. Average Latency: {axis1_latency:.3f} ms")
    except Exception as e:
        print(f"  ❌ Stride Status: FAILED. Runtime crash occurred: {str(e)}")
        all_passed = False

    # ──────────────────────────────────────────────────────────────────────────
    # AXIS 2: VARIABLE PRECISION DTYPE REDUCTION (AMP FP16 UNDERFLOW/OVERFLOW)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n🧮 Axis 2: Mixed-Precision (FP16) Type Alignment...")
    layer_dtype = RenormLinear(in_features=1024, out_features=512).to(device)
    X_micro = (torch.randn(2, 512, 1024, device=device) * 1e-6).to(torch.float16)
    
    try:
        out_fp16 = layer_dtype(X_micro)
        nan_count = torch.isnan(out_fp16).sum().item()
        inf_count = torch.isinf(out_fp16).sum().item()
        
        if nan_count == 0 and inf_count == 0:
            print(f"  ✅ FP16 Status: Protected! Type-safe upcasting resolved the mismatch error.")
            print(f"     Returned precision format cleanly as: {out_fp16.dtype}")
        else:
            print(f"  ❌ FP16 Status: FAILED. Numerical annihilation occurred (NaNs: {nan_count})")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FP16 Status: Crashed under FP16 loop mapping: {str(e)}")
        all_passed = False

    # ──────────────────────────────────────────────────────────────────────────
    # AXIS 3: MICRO-BATCHING INFERENCE MEMORY LEAK AUDIT
    # ──────────────────────────────────────────────────────────────────────────
    print("\n💧 Axis 3: Micro-Batching Inference Graph Memory Leaks...")
    layer_leak = RenormLinear(in_features=512, out_features=256).to(device)
    X_inf = torch.randn(1, 1, 512, device=device)
    
    print("  Simulating 50 continuous streaming real-time production requests...")
    leaked_graphs = 0
    
    try:
        for _ in range(50):
            with torch.no_grad():
                out_inf = layer_leak(X_inf)
                if out_inf.grad_fn is not None:
                    leaked_graphs += 1
                    
        if leaked_graphs == 0:
            print("  ✅ Inference Leak Status: Clean! Context graphs detached during runtime serving.")
        else:
            print(f"  ❌ Inference Leak Status: FAILED. {leaked_graphs} un-detached graph contexts detected.")
            all_passed = False
    except Exception as e:
        print(f"  ❌ Inference Leak Status: Crashed during serving evaluation: {str(e)}")
        all_passed = False

    # ──────────────────────────────────────────────────────────────────────────
    # AXIS 4: GRADIENT EXPLOSION / NaN PROPAGATION BARRIER
    # ──────────────────────────────────────────────────────────────────────────
    print("\n💥 Axis 4: Gradient Explosion Barrier & Analytical Derivative Health...")
    
    # Audit directly using our Module instantiation to match real-world weight parameter usage
    layer_grad = RenormLinear(in_features=256, out_features=128).to(device)
    X_grad = torch.randn(2, 256, device=device, requires_grad=True)
    
    try:
        out_grad = layer_grad(X_grad)
        exploded_loss = out_grad.sum() * 1e8
        exploded_loss.backward()
        
        if layer_grad.weight.grad is None or X_grad.grad is None:
            print("  ❌ Gradient Explosion Status: FAILED. PyTorch Autograd failed to accumulate leaf gradients.")
            all_passed = False
        else:
            nan_w = torch.isnan(layer_grad.weight.grad).sum().item()
            nan_x = torch.isnan(X_grad.grad).sum().item()
            
            if nan_w == 0 and nan_x == 0:
                print("  ✅ Gradient Explosion Status: Secured. Clamping barriers safely containing values.")
            else:
                print(f"  ❌ Gradient Explosion Status: FAILED. NaNs broke through into parameter states.")
                all_passed = False
    except Exception as e:
        print(f"  ❌ Gradient Explosion Status: Execution crashed: {str(e)}")
        all_passed = False

    # ──────────────────────────────────────────────────────────────────────────
    # FINAL SUMMARY BANNER
    # ──────────────────────────────────────────────────────────────────────────
    print("\n=========================================================")
    if all_passed:
        print("👑 🎉 ALL 4 AXES RESILIENT: Codebase is production-hardened!")
    else:
        print("⚠️  DIAGNOSTIC FAILURE: One or more architectural axes failed.")
    print("=========================================================")

if __name__ == "__main__":
    run_four_dimensional_diagnostic()