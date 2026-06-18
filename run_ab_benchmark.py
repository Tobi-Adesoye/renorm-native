import time

import torch
import torch.nn as nn

from renorm.layers import RenormTransformerLayer

# -----------------------------
# CONFIG
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4
SEQ = 16
DIM = 512
STEPS = 50


# -----------------------------
# DATA
# -----------------------------
def make_input():
    return torch.randn(BATCH, SEQ, DIM, device=device, requires_grad=True)


# -----------------------------
# MODEL A (BASELINE)
# -----------------------------
def build_baseline():
    return nn.TransformerEncoderLayer(d_model=DIM, nhead=8, batch_first=True).to(device)


# -----------------------------
# MODEL B (RENORM)
# -----------------------------
def build_renorm():
    return RenormTransformerLayer(dim=DIM, heads=8).to(device)


# -----------------------------
# BENCHMARK RUNNER
# -----------------------------
def benchmark(model, name):
    model.train()

    # warmup
    for _ in range(5):
        x = make_input()
        y = model(x)
        loss = y.mean()
        loss.backward()

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    mem_before = torch.cuda.memory_allocated() if device == "cuda" else 0

    for _ in range(STEPS):
        x = make_input()

        y = model(x)
        loss = y.mean()

        loss.backward()

        if hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    mem_after = torch.cuda.memory_allocated() if device == "cuda" else 0

    print("\n==============================")
    print(f"MODEL: {name}")
    print(f"Time total: {end - start:.4f}s")
    print(f"Avg step: {(end - start) / STEPS * 1000:.3f} ms")
    print(f"Memory delta: {(mem_after - mem_before) / 1e6:.2f} MB")
    print("==============================\n")


# -----------------------------
# RUN A/B TEST
# -----------------------------
if __name__ == "__main__":
    print(f"Running on device: {device}")

    print("\n--- BASELINE ---")
    benchmark(build_baseline(), "PyTorch Transformer")

    print("\n--- RENORM ---")
    benchmark(build_renorm(), "Renorm Transformer")
