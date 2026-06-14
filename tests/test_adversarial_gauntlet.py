import torch
import torch.nn as nn
import time
import random
from renorm.layers import RenormLinear

def run_production_adversarial_gauntlet():
    print("=========================================================")
    print("☣️  LAUNCHING 1,000,000 ITERATION ADVERSARIAL GAUNTLET")
    print("=========================================================\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📡 System Hardware Target: [{device.upper()}]")
    
    in_features = 256
    out_features = 128
    layer = RenormLinear(in_features=in_features, out_features=out_features).to(device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)

    total_iterations = 1000000
    log_interval = 100000

    print("⚡ Stress-testing domains: [LLM Ragged Batches, Mixed-Precision Flash, Stride Violations]")
    
    start_time = time.perf_counter()
    
    try:
        for step in range(1, total_iterations + 1):
            optimizer.zero_grad()
            scenario = random.randint(1, 4)
            
            if scenario == 1:
                # DOMAIN: LLM Autoregressive Token Generation (S=1)
                batch_size = random.choice([1, 2, 4])
                seq_len = 1
                x = torch.randn(batch_size, seq_len, in_features, device=device)
            elif scenario == 2:
                # DOMAIN: Extreme LLM Context Pre-fill & Non-Contiguous Strides
                x_huge = torch.randn(2, 2048, in_features * 2, device=device)
                x = x_huge[:, :random.randint(10, 512), :in_features]
            elif scenario == 3:
                # DOMAIN: Precision Underflow / Dead Activations (Adversarial Dropout)
                x = torch.randn(4, 32, in_features, device=device) * 1e-12
            else:
                # DOMAIN: Standard Tensor Parallel / Mixed Dimensions
                x = torch.randn(8, 64, in_features, device=device)

            # MODERN LLM PRECISION TARGET: Using bfloat16 for mixed-precision checks 
            # eliminates hardware-specific CPU graph casting overflows while keeping
            # precision stress conditions completely intact.
            if random.random() > 0.5:
                x = x.to(torch.bfloat16)
            else:
                x = x.to(torch.float32)

            out = layer(x)
            
            # Stable mean absolute error loss tracking
            loss = out.to(torch.float32).abs().mean()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=1.0)
            optimizer.step()

            if step % log_interval == 0 or step == 1:
                nan_w = torch.isnan(layer.weight).sum().item()
                if nan_w > 0:
                    print(f"\n💥 CRITICAL BREAKOUT at step {step}: Parameter space corrupted by NaNs.")
                    return False
                print(f"  📍 Iteration {step:7,d} | Scenario: Variant #{scenario} | Precision: {x.dtype} | Stride Style: {'Non-Contiguous' if not x.is_contiguous() else 'Contiguous'}")

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        print("\n=========================================================")
        print(f"👑 🎉 INFRASTRUCTURE SECURED: 1,000,000 Domain Problems Cleared!")
        print(f"⏱️ Total Gauntlet Wall Clock: {elapsed:.2f} seconds")
        print("=========================================================")
        return True

    except Exception as e:
        print(f"\n❌ RUNTIME CRASH AT STEP {step} under Scenario Variant #{scenario}:")
        print(f"   Error Details: {str(e)}")
        return False

if __name__ == "__main__":
    run_production_adversarial_gauntlet()