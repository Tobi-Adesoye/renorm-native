import gc
import time

import torch
import torch.nn as nn


# =====================================================================
# 1. MOCK PRODUCTION BACKEND
# =====================================================================
class MockRenormBackend:
    """
    Simulates the pointer compaction and mathematical anchoring layers
    of the custom renorm-native C++/Triton compilation pipeline.
    """

    @staticmethod
    def native_stride_compact(x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous():
            return x.contiguous()
        return x

    @staticmethod
    def anchor_variance_backward_graph(x: torch.Tensor, threshold: float = 1e4) -> torch.Tensor:
        with torch.no_grad():
            var = torch.var(x)

        if var > threshold:
            alpha = 0.01
            mean = torch.mean(x)
            return (x - mean) / torch.sqrt(var * alpha + 1e-5)

        return x

    @staticmethod
    def fused_linear_forward(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, weight, bias)


backend = MockRenormBackend()


# =====================================================================
# 2. CORE ARCHITECTURAL EXPORT: RenormLinear
# =====================================================================
class RenormLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, variance_anchor: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_anchor = variance_anchor

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = backend.native_stride_compact(x)

        if self.use_anchor and self.training:
            x = backend.anchor_variance_backward_graph(x, threshold=1e4)

        return backend.fused_linear_forward(x, self.weight, self.bias)


# =====================================================================
# 3. PRODUCTION STRESS TEST ENGINE
# =====================================================================
def run_stress_test(layer_factory, batch_sizes, dim=4096):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    gc.collect()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> Target Compute Hardware Node Routed to: {device.upper()}")

    layer = layer_factory(dim).to(device)
    layer.train()

    start_time = time.time()

    try:
        for i in range(50):
            dynamic_seq_len = batch_sizes[i % len(batch_sizes)]

            x = torch.randn(dynamic_seq_len, dim, device=device, requires_grad=True) * 150.0

            out = layer(x)

            loss = (out**2).sum() * 1e-4
            loss.backward()

            with torch.no_grad():
                for param in layer.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            raise RuntimeError("Catastrophic NaN Leakage Detected!")

                        param.data -= 0.001 * param.grad
                        param.grad.zero_()

        if device == "cuda":
            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            peak_mem = (dim * dim * 4 * 2) / (1024**2)

        duration = time.time() - start_time

        return {
            "status": "SUCCESS",
            "peak_vram_mb": f"{peak_mem:.2f} MB",
            "duration_sec": f"{duration:.4f}s",
        }

    except RuntimeError as e:
        if "Out of memory" in str(e):
            return {
                "status": "OOM_CRASH",
                "peak_vram_mb": "CRITICAL_LIMIT",
                "duration_sec": "FAILED",
            }
        raise e


# =====================================================================
# 4. EXECUTION MATRIX
# =====================================================================
def pytorch_factory(d):
    return torch.nn.Linear(d, d)


def renorm_factory(d):
    return RenormLinear(d, d, variance_anchor=True)


if __name__ == "__main__":
    ragged_batch_profiles = [32, 128, 16, 256, 64, 512, 8, 1024]
    hidden_dimension = 4096

    print("=" * 70)
    print("RUNNING WORKLOAD AUDIT: STANDARD PYTORCH LINEAR (BASELINE)")
    print("=" * 70)

    baseline_results = run_stress_test(pytorch_factory, ragged_batch_profiles, hidden_dimension)
    print(f"Execution Metric Report: {baseline_results}\n")

    print("=" * 70)
    print("RUNNING WORKLOAD AUDIT: RENORM-NATIVE INFRASTRUCTURE LAYER")
    print("=" * 70)

    optimized_results = run_stress_test(renorm_factory, ragged_batch_profiles, hidden_dimension)
    print(f"Execution Metric Report: {optimized_results}")
    print("=" * 70)
