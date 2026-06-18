import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim

from renorm.transformer import RenormTransformerLayer


def execute_convergence_run():
    print("=== STARTING MULTI-EPOCH CONVERGENCE BENCHMARK ===")

    # 1. Setup deep network topology
    NUM_LAYERS = 12
    BATCH_SIZE = 4
    SEQ_LEN = 16
    EMBED_DIM = 64
    HEADS = 4
    EPOCHS = 50

    # Build a sequential model wrapping our custom layers
    model = nn.Sequential(
        *[RenormTransformerLayer(d_model=EMBED_DIM, n_heads=HEADS) for _ in range(NUM_LAYERS)]
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()

    # Generate fixed, non-trivial synthetic dataset targets
    # This simulates a complex mapping from an input sequence space to a target manifold
    X_train = torch.randn(10, BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    Y_train = torch.sin(X_train) * 0.5 + torch.cos(X_train) * 0.3  # Non-linear objective

    print(f"Tracking 24 Independent Bounded Manifolds Across {EPOCHS} Epochs...")
    print("-" * 80)
    print(f"{'Epoch':<8} | {'Current Loss':<14} | {'Layer 0 Attn β':<16} | {'Layer 11 MLP β':<16}")
    print("-" * 80)

    # 2. Training Loop Progression
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0

        for batch_idx in range(len(X_train)):
            optimizer.zero_grad()

            inputs = X_train[batch_idx]
            targets = Y_train[batch_idx]

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            # Apply standard gradient clipping to prevent optimizer step destabilization
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Log telemetry metrics every 10 epochs, and the final epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            avg_loss = epoch_loss / len(X_train)

            with torch.no_grad():
                # Extract the actual bounded beta scales via the sigmoid manifold
                layer_0_attn_beta = torch.sigmoid(model[0].attn.beta_proxy).item()
                layer_11_mlp_beta = torch.sigmoid(model[-1].beta_mlp_proxy).item()

            print(
                f"{epoch:<8} | {avg_loss:<14.6f} | {layer_0_attn_beta:<16.6f} | {layer_11_mlp_beta:<16.6f}"
            )

    print("-" * 80)
    print("=== CONVERGENCE PROFILE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    execute_convergence_run()
