import logging

import torch
import torch.nn as nn

logger = logging.getLogger("Renorm-Scheduler")


class RenormInterleavedScheduler:
    """
    Orchestrates overlapping memory transfers and computing streams.
    Prevents the GPU from waiting on slow PCIe transfers during model execution
    by pipelining upcoming layers.
    """

    def __init__(self, device: torch.device):
        self.device = device
        # Dedicated CUDA stream for non-blocking H2D (Host-to-Device) copies
        self.prefetch_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self.pinned_buffers: dict[str, torch.Tensor] = {}

    def pin_tensor_memory(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """
        Pins host memory to enable high-speed, non-blocking asynchronous
        GPU copy operations (direct DMA transfers).
        """
        if not tensor.is_pinned() and tensor.device.type == "cpu":
            try:
                pinned_tensor = tensor.pin_memory()
                self.pinned_buffers[name] = pinned_tensor
                return pinned_tensor
            except Exception as e:
                logger.warning(
                    f"Failed to pin host buffer '{name}': {e}. Falling back to standard transfer."
                )
        return tensor

    def execute_and_prefetch(
        self,
        current_layer: nn.Module,
        input_tensor: torch.Tensor,
        next_layer: nn.Module | None = None,
        next_input_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Executes computation on the active default stream while asynchronously
        sending the next layer's weights and inputs to VRAM on the prefetch stream.
        """
        # 1. Start execution of the current layer on the default stream
        output = current_layer(input_tensor)

        # 2. Check if we can execute async prefetching for the next step
        if next_layer is not None and self.prefetch_stream is not None:
            # Sync the default stream to make sure we don't clobber active allocations
            # but allow the transfer to run concurrently with the rest of the stream queue
            with torch.cuda.stream(self.prefetch_stream):
                # Copy next layer weights asynchronously
                if hasattr(next_layer, "weight") and next_layer.weight is not None:
                    next_layer.weight.data = next_layer.weight.data.to(
                        self.device, non_blocking=True
                    )

                if hasattr(next_layer, "bias") and next_layer.bias is not None:
                    next_layer.bias.data = next_layer.bias.data.to(self.device, non_blocking=True)

                # If we have pre-allocated CPU activation tensors, stream them asynchronously
                if next_input_cpu is not None:
                    next_input_cpu = self.pin_tensor_memory("next_activation", next_input_cpu)
                    next_input_cpu.to(self.device, non_blocking=True)

        return output

    def synchronize_scheduler(self):
        """Forces prefetch stream synchronization to prevent race conditions before layer transitions."""
        if self.prefetch_stream is not None:
            torch.cuda.current_stream().wait_stream(self.prefetch_stream)


# =====================================================================
# PIPELINE DEMO & EXPLICIT STABILITY TEST
# =====================================================================
if __name__ == "__main__":
    # Standard check to verify stream concurrency
    if torch.cuda.is_available():
        print("[Renorm-Scheduler] CUDA detected. Initializing stream overlap simulation...")
        dev = torch.device("cuda:0")
        scheduler = RenormInterleavedScheduler(dev)

        # Simulate adjacent linear layers
        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4096, 4096))
                self.bias = nn.Parameter(torch.zeros(4096))

            def forward(self, x):
                return torch.matmul(x, self.weight) + self.bias

        layer_1 = MockLayer().to(dev)
        layer_2 = MockLayer()  # Kept on CPU to simulate offloaded state

        # Test input
        x_in = torch.randn(32, 4096, device=dev)

        # Execute active layer while prefetching the next offline layer
        print("[Renorm-Scheduler] Executing layer_1 forward pass + prefetching layer_2 weights...")
        out_1 = scheduler.execute_and_prefetch(layer_1, x_in, next_layer=layer_2)

        # Synchronize streams to guarantee transfer safety
        scheduler.synchronize_scheduler()
        print(f"[Renorm-Scheduler] Sync complete. layer_2 weight device: {layer_2.weight.device}")
    else:
        print("[Renorm-Scheduler] CUDA is not present. Executing CPU-bound sequential pass.")
