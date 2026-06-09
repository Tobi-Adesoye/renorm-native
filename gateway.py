import os
import sys
import json

class RenormKernelGateway:
    """
    Plug-and-Play System Orchestration Gateway Middleware.
    Completely self-bootstrapping, zero-dependency on import, and framework-agnostic.
    Auto-generates localized configuration rule profiles if missing.
    """
    def __init__(self, runtime_flags: list = None):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.flags = runtime_flags or sys.argv
        self.config_path = os.path.join(self.root_dir, "gateway_profiles.json")
        
        # 1. Self-Bootstrapping: Auto-generate configuration matrix if missing
        if not os.path.exists(self.config_path):
            self._bootstrap_default_config()
            
        self.config = self._load_config()
        self.hw_name, self.active_hw = self._autodetect_hardware()
        self._state_holder = {"active_buffer": None}

    def _bootstrap_default_config(self):
        """Generates the externalized rule profile matrix automatically on zero-config clean runs."""
        default_matrix = {
            "hardware_profiles": {
                "amd_rocm": {"cache_line_bytes": 128, "preferred_alignment": 32, "max_sram_head": 8},
                "nvidia_cuda": {"cache_line_bytes": 128, "preferred_alignment": 32, "max_sram_head": 8},
                "generic_cpu": {"cache_line_bytes": 64, "preferred_alignment": 4, "max_sram_head": 0}
            },
            "runtime_rules": [
                {"flag": "--use-renorm", "action": "ENGAGE_GATEWAY"},
                {"flag": "--use-flash-attention", "action": "SET_FALLBACK_ATTN"}
            ]
        }
        with open(self.config_path, "w") as f:
            json.dump(default_matrix, f, indent=2)

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception:
            # Emergency inline fallback if JSON is corrupted by a user edit
            return {
                "hardware_profiles": {"generic_cpu": {"cache_line_bytes": 64, "preferred_alignment": 4, "max_sram_head": 0}},
                "runtime_rules": [{"flag": "--use-renorm", "action": "ENGAGE_GATEWAY"}]
            }

    def _autodetect_hardware(self) -> tuple[str, dict]:
        """Soft-detects underlying accelerator fabrics without crashing if framework imports are broken."""
        try:
            import torch
            if hasattr(torch, "version") and torch.version.hip is not None:
                return "amd_rocm", self.config["hardware_profiles"].get("amd_rocm")
            if torch.cuda.is_available():
                return "nvidia_cuda", self.config["hardware_profiles"].get("nvidia_cuda")
        except ImportError:
            pass # Gracefully handle non-PyTorch environments
            
        return "generic_cpu", self.config["hardware_profiles"].get("generic_cpu")

    def process_graph(self, layer_dims: dict, sequence_len: int, batch_size: int = 1) -> dict:
        """Evaluates tensor topologies dynamically against the active rule configuration profile."""
        dim = layer_dims.get("dim", 512)
        heads = layer_dims.get("heads", 1)
        head_dim = dim // heads if heads > 0 else dim
        
        triggered_actions = [rule["action"] for rule in self.config["runtime_rules"] if rule["flag"] in self.flags]
        row_bytes = dim * 4
        is_coalesced = (row_bytes % self.active_hw.get("cache_line_bytes", 64)) == 0
        intensity = "MEMORY_BOUND_DECODE" if sequence_len == 1 else "COMPUTE_BOUND_PREFILL"

        manifest = {
            "status": "OPERATIONAL" if "ENGAGE_GATEWAY" in triggered_actions else "BYPASS_MODE",
            "hardware_target": self.hw_name,
            "telemetry": {
                "intensity_mode": intensity,
                "memory_coalesced": is_coalesced,
                "active_directives_count": len(triggered_actions)
            },
            "execution_directives": []
        }

        if "ENGAGE_GATEWAY" not in triggered_actions:
            manifest["execution_directives"].append("PASSTHROUGH_TO_RAW_BACKEND")
            return manifest

        # Apply alignment paddings
        if not is_coalesced or self.hw_name == "amd_rocm":
            pad = self.active_hw.get("preferred_alignment", 32)
            manifest["execution_directives"].append(f"ENFORCE_STRIDE_ALIGNMENT_PAD_{pad}")

        # Process micro-head fencing
        if head_dim <= self.active_hw.get("max_sram_head", 0):
            manifest["execution_directives"].append("EXECUTE_SRAM_REGISTERS_FUSED_KERNEL")

        # Allocation anchoring guard
        try:
            import torch
            if torch.cuda.is_available() or self.hw_name == "amd_rocm":
                expected_el = sequence_len * dim
                curr_buf = self._state_holder.get("active_buffer")
                if curr_buf is None or curr_buf.shape[0] != expected_el:
                    self._state_holder["active_buffer"] = torch.zeros(expected_el, device="cuda" if torch.cuda.is_available() else "cpu")
                    manifest["execution_directives"].append("ANCHOR_PERSISTENT_STATE_TRACKING_BUFFER")
        except ImportError:
            manifest["execution_directives"].append("SKIP_TORCH_BUFFER_ANCHOR_NON_PYTORCH_ENV")

        return manifest

    @staticmethod
    def inject(target_module, hook_method_name: str):
        """
        Static plug-and-play hook tool. Wraps any framework execution entry-point 
        directly upon reference.
        """
        original_method = getattr(target_module, hook_method_name, None)
        if not original_method:
            return
            
        gateway_instance = RenormKernelGateway()

        def plug_and_play_wrapper(*args, **kwargs):
            # Inspect first argument shape dynamically if matching standard tensor profiles
            if args and hasattr(args[0], "shape") and len(args[0].shape) >= 2:
                tensor_shape = args[0].shape
                # Infer dynamic dimensions: (batch, seq, dim) or (seq, dim)
                dim = tensor_shape[-1]
                seq = tensor_shape[-2] if len(tensor_shape) >= 2 else 1
                gateway_instance.process_graph({"dim": dim}, sequence_len=seq)
            return original_method(*args, **kwargs)

        setattr(target_module, hook_method_name, plug_and_play_wrapper)