import statistics
import time

import torch
import torch.nn as nn

from renorm.layers import RenormTransformerLayer

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cpu")
RUNS = 10
WARMUP = 3
SEQ_LEN = 32
DIM = 512


# =========================================================
# MODELS
# =========================================================
def build_baseline():
    return nn.TransformerEncoderLayer(d_model=DIM, nhead=8, batch_first=True)


def build_renorm():
    return RenormTransformerLayer(dim=DIM, heads=8)


# =========================================================
# BENCHMARK CORE
# =========================================================
def benchmark(model_builder, name: str):
    times = []

    model = model_builder().to(DEVICE)
    model.eval()

    x = torch.randn(2, SEQ_LEN, DIM, device=DEVICE)

    # -------------------
    # warmup (VERY IMPORTANT even on CPU)
    # -------------------
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)

    # -------------------
    # timed runs
    # -------------------
    with torch.no_grad():
        for _ in range(RUNS):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return {
        "name": name,
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n==============================")
    print("CPU-ONLY A/B BENCHMARK SUITE")
    print("==============================")
    print(f"Device: {DEVICE}")
    print(f"Runs: {RUNS} | Warmup: {WARMUP}")
    print("==============================\n")

    baseline = benchmark(build_baseline, "PyTorch Transformer")
    renorm = benchmark(build_renorm, "Renorm Transformer")

    def print_report(r):
        print(f"--- {r['name']} ---")
        print(f"Mean: {r['mean_ms']:.3f} ms")
        print(f"Std : {r['std_ms']:.3f} ms")
        print(f"Min : {r['min_ms']:.3f} ms")
        print(f"Max : {r['max_ms']:.3f} ms\n")

    print_report(baseline)
    print_report(renorm)

    speedup = baseline["mean_ms"] / renorm["mean_ms"]

    print("==============================")
    print("FINAL COMPARISON")
    print("==============================")
    print(f"Speedup (baseline / renorm): {speedup:.3f}x")
    print("==============================\n")


if __name__ == "__main__":
    main()
