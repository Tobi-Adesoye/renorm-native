import sys

# Point Python at your local renorm source tree
sys.path.insert(0, r"C:\Users\my pc\Desktop\renorm-native\src")

import torch

from renorm import RenormLinear


def test_zero_padded_input():
    print("=" * 60)
    print("🔬 MXFP4 / Zero-Padded Tensor Stability Validation")
    print("=" * 60)

    hidden_dim = 4096

    # Simulate hidden states similar to an LLM activation tensor
    x = torch.randn(1, 128, hidden_dim)

    # Inject zero-padded regions to emulate sparse or padded tokens
    x[:, 10:20, :] = 0.0

    print("⚠️ Injected zero-padded slices into the activation tensor.")

    # Instantiate your Renorm layer
    layer = RenormLinear(
        in_features=hidden_dim,
        out_features=hidden_dim,
    )

    # Forward pass
    y = layer(x)

    # Verify numerical stability
    nan_count = torch.isnan(y).sum().item()
    inf_count = torch.isinf(y).sum().item()

    print(f"Output shape : {tuple(y.shape)}")
    print(f"NaN count    : {nan_count}")
    print(f"Inf count    : {inf_count}")

    assert torch.isfinite(y).all(), "Output contains NaN or Inf!"

    print()
    print("✅ SUCCESS: Forward pass completed successfully.")
    print("✅ No NaN values detected.")
    print("✅ No Inf values detected.")
    print("=" * 60)


if __name__ == "__main__":
    test_zero_padded_input()
