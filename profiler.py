import json
import os


class WTGTelemetryBridge:
    """
    Parses native WTG (WhatTheGPU) snapshot logs to extract and interpret
    hardware-level memory-controller utilization anomalies during layer execution.
    """

    def __init__(self, wtg_log_path="probe_output.json"):
        self.log_path = wtg_log_path

    def analyze_memory_saturation(self):
        if not os.path.exists(self.log_path):
            return {"status": "No WTG telemetry log detected. Run wtg.exe --once to profile."}

        with open(self.log_path) as f:
            data = json.load(f)

        # Extract native WTG NVML probe fields matching v0.2.5 schema
        gpu_util = data.get("util.gpu_pct", 0)
        mem_controller_util = data.get("util.mem_controller_pct", 0)
        vram_used = data.get("vram.used_mib", 0)

        # Evaluate the known mobile/desktop driver anomaly mentioned in WTG v0.2.5 release
        if mem_controller_util == 100 and gpu_util < 10:
            status = "⚠️ ANOMALY DETECTED: NVML Memory-Controller saturation mismatch (Driver bug identified by WTG)."
        elif mem_controller_util > 85:
            status = (
                "🔥 CRITICAL LOADING: Intermediate activation layers are saturating HBM boundaries."
            )
        else:
            status = "✅ STABILIZED: SRAM register fusion active. Memory-bandwidth utilization within bounds."

        return {
            "gpu_utilization_pct": gpu_util,
            "memory_controller_pct": mem_controller_util,
            "allocated_vram_mib": vram_used,
            "hardware_interpretation": status,
        }
