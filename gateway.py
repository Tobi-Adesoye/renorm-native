import os
import sys
import torch
import json

class RenormKernelGateway:
    """
    All-Encompassing Declarative Topology, Boundary & Environment Resolution Engine.
    Abstracts hardware platforms into core physical constraints and handles absolute path 
    resolution to remain fully operational across symlink-free local installations.
    """
    def __init__(self, hardware_profile: dict = None, runtime_flags: list = None):
        # 1. Resolve absolute environment footprint paths (Bypasses symlink/editable installation dependencies)
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Parse operational execution flags passed by client applications
        flags = runtime_flags or sys.argv
        self.use_renorm_active = "--use-renorm" in flags
        self.use_flash_attn_fallback = "--use-flash-attention" in flags

        # 3. Derive the hardware capability matrix based on current active metal
        if hardware_profile:
            self.hw = hardware_profile
        else:
            is_rocm = torch.version.hip is not None if hasattr(torch, "version") else False
            self.hw = {
                "name": "rocm_hip_amd" if is_rocm else "discrete_cuda",
                "cache_line_bytes": 128,
                "has_unified_memory": False,
                "preferred_alignment_elements": 32, 
                "max_sram_head_dim": 8
            }
            
        self._state_holder = {
            "active_buffer": None,
            "fencing_active": False,
            "stride_padding_applied": False
        }

    def _analyze_tensor_properties(self, layer_dims: dict, sequence_len: int, batch_size: int) -> dict:
        """Extracts immutable mathematical properties from the tensor graph layout."""
        dim = layer_dims.get("dim", 512)
        heads = layer_dims.get("heads", 1)
        head_dim = dim // heads if heads > 0 else dim
        bytes_per_element = 4
        
        row_bytes = dim * bytes_per_element
        is_coalesced = (row_bytes % self.hw["cache_line_bytes"]) == 0
        
        if sequence_len == 1:
            intensity = "MEMORY_BOUND_DECODE"
        elif batch_size * sequence_len < 1024:
            intensity = "MEMORY_BOUND_SMALL_BATCH"
        else:
            intensity = "COMPUTE_BOUND_PREFILL"

        return {
            "dim": dim,
            "head_dim": head_dim,
            "is_coalesced": is_coalesced,
            "row_bytes": row_bytes,
            "intensity": intensity,
            "expected_elements": sequence_len * dim
        }

    def process_graph(self, layer_dims: dict, sequence_len: int, batch_size: int = 1) -> dict:
        """Evaluates tensor profiles and issues execution directives safely relative to resolved root paths."""
        props = self._analyze_tensor_properties(layer_dims, sequence_len, batch_size)
        
        manifest = {
            "status": "OPERATIONAL" if self.use_renorm_active else "BYPASS_MODE",
            "hardware_target": self.hw["name"],
            "environment_root": self.root_dir,
            "telemetry": {
                "intensity_mode": props["intensity"],
                "memory_coalesced": props["is_coalesced"],
                "renorm_flag_engaged": self.use_renorm_active
            },
            "execution_directives": []
        }

        if not self.use_renorm_active:
            manifest["execution_directives"].append("PASSTHROUGH_TO_RAW_BACKEND_WITHOUT_RE_ALIGNMENT")
            return manifest

        # Guard A: Cache Line Stride Alignment Padding
        if not props["is_coalesced"] or self.hw["name"] == "rocm_hip_amd":
            self._state_holder["stride_padding_applied"] = True
            manifest["execution_directives"].append(f"ENFORCE_STRIDE_ALIGNMENT_PAD_{self.hw['preferred_alignment_elements']}")

        # Guard B: Nano-Scale Hardware Fencing
        if props["head_dim"] <= self.hw["max_sram_head_dim"]:
            self._state_holder["fencing_active"] = True
            manifest["execution_directives"].append("EXECUTE_SRAM_REGISTERS_FUSED_KERNEL")

        # Guard C: Stateful Allocation Anchor Pool
        expected_el = props["expected_elements"]
        curr_buf = self._state_holder["active_buffer"]
        if curr_buf is None or curr_buf.shape[0] != expected_el:
            if torch.cuda.is_available() or self.hw["name"] == "rocm_hip_amd":
                self._state_holder["active_buffer"] = torch.zeros(expected_el, device="cuda" if torch.cuda.is_available() else "cpu")
            manifest["execution_directives"].append("ANCHOR_PERSISTENT_STATE_TRACKING_BUFFER")

        return manifest

if __name__ == "__main__":
    print("--- 🛰️ EVALUATING ARCHITECTURAL LOCATION & FLAGS ---")
    mock_cli_args = ["main.py", "--use-renorm"]
    
    gateway = RenormKernelGateway(runtime_flags=mock_cli_args)
    manifest_out = gateway.process_graph({"dim": 111, "heads": 3}, sequence_len=1)
    
    print(json.dumps(manifest_out, indent=2))