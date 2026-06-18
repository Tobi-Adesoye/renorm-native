import unittest

import torch

from renorm.layers import RenormBlock, RenormLinear


class TestForwardIntegrity(unittest.TestCase):
    def test_dimension_alignment(self):
        """Ensure input configurations snap correctly to hardware-optimal tile shapes."""
        layer = RenormLinear(in_features=11, out_features=23, hardware_alignment=16)
        self.assertEqual(layer.in_features, 16)
        self.assertEqual(layer.out_features, 32)

    def test_tensor_flow_shape(self):
        """Verify tensor dimensions remain invariant across the block transformation step."""
        batch_size, dim = 4, 32
        x = torch.randn(batch_size, dim)
        block = RenormBlock(dim=dim)

        out = block(x)
        self.assertEqual(out.shape, x.shape)

        # Ensure gradients flow backwards through the learnable parameters cleanly
        loss = out.mean()
        loss.backward()
        self.assertIsNotNone(block.beta.grad)
        self.assertIsNotNone(block.fn.weight.grad)


if __name__ == "__main__":
    unittest.main()
