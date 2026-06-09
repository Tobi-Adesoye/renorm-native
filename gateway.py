import sys

class RenormKernelGateway:
    """
    Applies the Reddit BM25 engineering rule: Uses structured keyword/schema
    matching rather than loose semantic embeddings to route operational payloads
    to target hardware compilation modes.
    """
    def __init__(self, hardware_context: str):
        self.hardware_context = hardware_context.lower()

    def determine_kernel_route(self, layer_dims: dict) -> str:
        # Flattened property projections acting as deterministic routing rules
        schema_keywords = {
            "triton": ["dim=512", "heads=8", "sram_fusion", "cuda"],
            "rocm": ["gfx1100", "hip", "amd", "gtt"],
            "standard_cpu": ["cpu", "emulation", "fallback"]
        }
        
        # Build an index payload string out of incoming dimensions
        payload_stream = f"device={self.hardware_context} " + " ".join([f"{k}={v}" for k, v in layer_dims.items()])
        
        # Lexical classification route selection
        for backend, keywords in schema_keywords.items():
            if any(keyword in payload_stream for keyword in keywords):
                return f"🚀 ROUTING MATRIX DIRECTLY TO: [{backend.upper()}] KERNEL ENGINES"
                
        return "🔄 ROUTING MATRIX TO: [STANDARD CPU AUTOGRAD EMULATION]"

if __name__ == "__main__":
    # Test our lexical gateway matching
    gateway = RenormKernelGateway(hardware_context="cpu")
    print(gateway.determine_kernel_route({"dim": 512, "heads": 8, "mode": "fallback"}))