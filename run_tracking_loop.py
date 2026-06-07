import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import torch, torch.nn as nn, torch.optim as optim
from renorm.containers import RenormContainer

def execute_drift_tracking():
    print("=== INITIALIZING PARAMETER DRIFT ENGINE (SIGMOID BOUNDED) ===")
    DEPTH, DIM, STEPS = 5, 64, 5
    model = RenormContainer(depth=DEPTH, dim=DIM)
    optimizer = optim.SGD(model.parameters(), lr=5.0) # Cranked learning rate to test boundary stress
    
    initial_betas = torch.sigmoid(model.beta_proxy).detach().clone()
    print("-" * 75)
    print(f"{'Step':<6} | {'Layer ID':<10} | {'Current Beta':<14} | {'Total Shift Delta':<18} | {'Proxy Value':<12}")
    print("-" * 75)
    
    for step in range(1, STEPS + 1):
        x, target = torch.randn(4, DIM), torch.randn(4, DIM)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            functional_betas = torch.sigmoid(model.beta_proxy)
            for layer_idx in [0, DEPTH - 1]:
                curr_beta = functional_betas[layer_idx].item()
                delta = curr_beta - initial_betas[layer_idx].item()
                proxy_val = model.beta_proxy[layer_idx].item()
                
                # Fixed: Resolve string calculation before inserting into the layout
                layer_label = f"Layer {layer_idx}"
                print(f"{step:<6} | {layer_label:<10} | {curr_beta:<14.6f} | {delta:<+18.6f} | {proxy_val:<12.4f}")
        print("-" * 75)

if __name__ == "__main__":
    execute_drift_tracking()