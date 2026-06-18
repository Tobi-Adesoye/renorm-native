import statistics
import time

import torch
import torch.nn as nn

from renorm.layers import RenormTransformerLayer

# =========================================================
# CONFIG (CPU ONLY)
# =========================================================
DEVICE = torch.device("cpu")

DIM = 512
HEADS = 8

SEQ_SIZES = [8, 16, 32, 64, 128, 256]
RUNS_PER_SIZE = 5


# =========================================================
# MODELS
# =========================================================
def baseline():
    return nn.TransformerEncoderLayer(d_model=DIM, nhead=HEADS, batch_first=True)


def renorm():
    return RenormTransformerLayer(dim=DIM, heads=HEADS)


# =========================================================
# CORE BENCHMARK
# =========================================================
def run_model(model, x):
    with torch.no_grad():
        start = time.perf_counter()
        _ = model(x)
        end = time.perf_counter()
    return (end - start) * 1000  # ms


def stress_test(model_builder, name: str):
    print("\n==============================")
    print(f"STRESS TEST: {name}")
    print("==============================")

    results = {}

    for seq in SEQ_SIZES:
        times = []

        model = model_builder().to(DEVICE)
        model.eval()

        x = torch.randn(2, seq, DIM, device=DEVICE)

        # warmup
        for _ in range(2):
            _ = model(x)

        for _ in range(RUNS_PER_SIZE):
            try:
                t = run_model(model, x)
                times.append(t)
            except Exception as e:
                print(f"[CRASH] seq={seq}: {str(e)}")
                times.append(None)

        valid = [t for t in times if t is not None]

        if len(valid) == 0:
            results[seq] = None
            print(f"Seq {seq}: FAILED (all runs crashed)")
            continue

        results[seq] = {
            "mean": statistics.mean(valid),
            "std": statistics.stdev(valid) if len(valid) > 1 else 0.0,
            "min": min(valid),
            "max": max(valid),
        }

        print(
            f"Seq {seq:<3} | "
            f"mean={results[seq]['mean']:.3f} ms | "
            f"std={results[seq]['std']:.3f} | "
            f"min={results[seq]['min']:.3f} | "
            f"max={results[seq]['max']:.3f}"
        )

    return results


# =========================================================
# COMPARISON REPORT
# =========================================================
def compare(baseline, renorm):
    print("\n==============================")
    print("SCALING COMPARISON")
    print("==============================")

    for seq in SEQ_SIZES:
        b = baseline.get(seq)
        r = renorm.get(seq)

        if not b or not r:
            print(f"Seq {seq}: incomplete")
            continue

        speedup = b["mean"] / r["mean"]

        print(
            f"Seq {seq:<3} | "
            f"speedup={speedup:.2f}x | "
            f"baseline={b['mean']:.3f}ms | "
            f"renorm={r['mean']:.3f}ms"
        )


# =========================================================
# MAIN
# =========================================================
def main():
    print("\n🚀 CPU STRESS TEST SUITE STARTING")
    print("Device:", DEVICE)
    print("Runs per config:", RUNS_PER_SIZE)

    baseline_results = stress_test(baseline, "PyTorch Transformer")
    renorm_results = stress_test(renorm, "Renorm Transformer")

    compare(baseline_results, renorm_results)

    print("\n✅ STRESS TEST COMPLETE")


if __name__ == "__main__":
    main()
