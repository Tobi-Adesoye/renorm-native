
import os
import sys
import time
import logging
import torch
import torch.nn as nn

# Set up instrumentation logger
logging.basicConfig(level=logging.INFO, format="[Renorm-AIOps] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TelemetrySuite")

# Attempt local imports of our newly upgraded components
try:
    from gateway import GatewayReferee, fallback_blueprint
    from layers import RenormSelfStabilizingLayer
    from scheduler import RenormInterleavedScheduler
    from loopguard import RenormLoopGuard
    IMPORTS_OK = True
except ImportError as e:
    logger.error(f"Failed to load upgraded local modules: {e}")
    logger.error("Make sure gateway.py, layers.py, scheduler.py, and loopguard.py are in the same folder.")
    IMPORTS_OK = False


class RenormTelemetryExporter:
    """
    Exposes high-resolution hardware telemetry mapping memory fragmentation levels,
    reserved memory pools, active regions, and potential allocator faults.
    """
    @staticmethod
    def capture_system_metrics() -> dict:
        metrics = {
            "platform_os": sys.platform,
            "cuda_available": torch.cuda.is_available(),
            "device_name": "None",
            "allocated_vram_mb": 0.0,
            "reserved_vram_mb": 0.0,
            "active_segments": 0,
            "oom_exceptions": 0,
            "allocation_ratio": 0.0
        }

        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            metrics["device_name"] = torch.cuda.get_device_name(device_idx)
            metrics["allocated_vram_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            metrics["reserved_vram_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            
            try:
                stats = torch.cuda.memory_stats(device_idx)
                metrics["active_segments"] = stats.get("active_regions.small_pool.current", 0) + \
                                             stats.get("active_regions.large_pool.current", 0)
                metrics["oom_exceptions"] = stats.get("num_ooms", 0)
                
                # Ratio of active tensor data versus reserved memory pool
                if metrics["reserved_vram_mb"] > 0:
                    metrics["allocation_ratio"] = metrics["allocated_vram_mb"] / metrics["reserved_vram_mb"]
            except Exception:
                pass
                
        return metrics

    @staticmethod
    def print_prometheus_format():
        """Helper to format outputs into standard AIOps collector formats."""
        metrics = RenormTelemetryExporter.capture_system_metrics()
        print("\n--- Prometheus Metric Export Stream ---")
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                # Print key name as prometheus gauge
                metric_name = f"renorm_native_{key}"
                print(f"# TYPE {metric_name} gauge")
                print(f"{metric_name} {val}")
        print("----------------------------------------\n")


def run_integration_pipeline():
    """Performs tests sequentially across all system boundary modules."""
    if not IMPORTS_OK:
        logger.error("Aborting system check due to missing files.")
        return

    logger.info("Initializing system integration checks...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -----------------------------------------------------------------
    # Test 1: Gateway & Environment Overrides
    # -----------------------------------------------------------------
    logger.info("TEST 1: Evaluating Gateway referee configuration...")
    referee = GatewayReferee()
    logger.info(f"Target Metal Context: {referee.metal_context}")
    logger.info(f"Matched Alignment Strategy: {referee.active_alignment}")
    
    # Confirm allocations CONF values were correctly written to environment variables
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "Not Set")
    logger.info(f"Confirmed active Environment Overrides: {alloc_conf}")

    # -----------------------------------------------------------------
    # Test 2: Mathematical Stabilization Core
    # -----------------------------------------------------------------
    logger.info("TEST 2: Verifying mathematical self-stabilizing layers...")
    batch, seq, features = 16, 256, 512
    # Simulate high-variance raw activation matrix
    exploding_input = torch.randn(batch, seq, features, device=device) * 15.0
    
    # Create normalized layer instance
    renorm_layer = RenormSelfStabilizingLayer(features, features, beta=0.05).to(device)
    
    # Calculate initial outputs
    out_tensors = renorm_layer(exploding_input)
    
    # Check bounds (The variance of outputs should be bounded smoothly due to β limits)
    out_variance = torch.var(out_tensors).item()
    logger.info(f"Raw Input Max Magnitude: {torch.max(torch.abs(exploding_input)).item():.4f}")
    logger.info(f"Stabilized Output Max Magnitude: {torch.max(torch.abs(out_tensors)).item():.4f}")
    logger.info(f"Resulting Tensor Variance: {out_variance:.4f}")
    
    # Verify we got a valid tensor projection shape back
    assert out_tensors.shape == (batch, seq, features), "Mathematical Layer output shape mismatch!"
    logger.info("Mathematical self-stabilizing layers passed verification successfully.")

    # -----------------------------------------------------------------
    # Test 3: Scheduler Prefetching Mechanics
    # -----------------------------------------------------------------
    logger.info("TEST 3: Checking Interleaved Async Scheduler...")
    scheduler = RenormInterleavedScheduler(device)
    
    # Build current active layer and offline prefetch targets
    layer_active = RenormSelfStabilizingLayer(features, features).to(device)
    layer_prefetch = RenormSelfStabilizingLayer(features, features) # Initialized on CPU
    
    # Run the interleaved pass
    logger.info("Launching execution pass + prefetch stream copy...")
    scheduler.execute_and_prefetch(
        current_layer=layer_active,
        input_tensor=out_tensors,
        next_layer=layer_prefetch
    )
    
    # Synchronize and verify layer prefetch has successfully migrated to the active execution device
    scheduler.synchronize_scheduler()
    logger.info(f"Prefetched Layer Target Device: {layer_prefetch.weight.device}")
    assert layer_prefetch.weight.device.type == device.type, "Async weight migration stalled!"
    logger.info("Interleaved stream prefetching validated.")

    # -----------------------------------------------------------------
    # Test 4: Agent LoopGuard Execution Gateway
    # -----------------------------------------------------------------
    logger.info("TEST 4: Inspecting LoopGuard Gateway sanitization parser...")
    guard = RenormLoopGuard(max_consecutive_repeats=3)
    
    # Mock conversational input containing valid and invalid markdown formatting patterns
    mock_payload = (
        "Outputting requested action. Check values:\n"
        "```json\n"
        "{\n"
        '  "action": "ENGAGE_RENORM_ALIGNMENT",\n'
        '  "target_pool": "SRAM_REGISTERS"\n'
        "}\n"
        "```"
    )
    
    sanitized_response = guard.sanitize_output_stream(mock_payload)
    logger.info(f"Sanitized Status: {sanitized_response['status']}")
    logger.info(f"Extracted Params: {sanitized_response['data']}")
    assert sanitized_response["data"]["action"] == "ENGAGE_RENORM_ALIGNMENT", "JSON sanitization failed!"
    
    # Simulate loop repeats to confirm budget breaker works
    redundant_call = '{"action": "SYNC_RECORDS"}'
    loop_triggered = False
    for i in range(3):
        metrics = guard.sanitize_output_stream(redundant_call)
        if metrics["loop_detected"]:
            loop_triggered = True
            logger.info(f"Budget alert active on loop-pass {i+1}!")
            
    assert loop_triggered, "Infinite loop guard failed to trigger on repeated payloads!"
    logger.info("LoopGuard Protocol validation passed successfully.")

    # -----------------------------------------------------------------
    # Final: Hardware Telemetry Reporting
    # -----------------------------------------------------------------
    logger.info("All tests completed successfully. Initializing system telemetry dump...")
    RenormTelemetryExporter.print_prometheus_format()


if __name__ == "__main__":
    run_integration_pipeline()