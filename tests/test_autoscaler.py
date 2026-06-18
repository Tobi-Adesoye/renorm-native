import unittest

import torch

from renorm.containers import RenormContainer


class TestAutoScalerPerformance(unittest.TestCase):
    def test_production_scale_stability(self):
        """
        Evaluates the Auto-Scaler across a simulated production-scale layout.
        Verifies that variance mapping preserves gradient signals at extreme depths.
        """
        PRODUCTION_DEPTH = 300
        PRODUCTION_DIM = 512  # Increased dimension to test scaling bounds

        # Initialize the self-regulating deep container
        model = RenormContainer(depth=PRODUCTION_DEPTH, dim=PRODUCTION_DIM)

        x = torch.randn(1, PRODUCTION_DIM, requires_grad=True)
        out = model(x)

        # Execute the backward sweep
        loss = out.mean()
        loss.backward()

        grad_var = x.grad.var().item()

        print("\n" + "=" * 20 + " AUTO-SCALER RUNTIME VERIFICATION " + "=" * 20)
        print(
            f" Context Settings   : Depth={PRODUCTION_DEPTH} | Feature Dimensions={PRODUCTION_DIM}"
        )
        print(f" Evaluated Initial Beta Heuristic Value : {model.beta[0].item():.6f}")
        print(f" Captured Outbound Gradient Variance    : {grad_var:.6f}")
        print("=" * 74 + "\n")

        # Ensure the gradient hasn't vanished (0.0) or detonated (NaN/Inf)
        self.assertTrue(
            grad_var > 1e-7,
            "Signal has dropped below operational visibility limits (Vanished).",
        )
        self.assertTrue(grad_var < 1.0, "Signal has exceeded stable scale bounds (Exploded).")


if __name__ == "__main__":
    unittest.main()
