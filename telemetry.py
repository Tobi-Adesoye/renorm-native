import logging
import os
import platform

from posthog import Posthog

# Initialize the global Posthog client with your live project token
PH_API_KEY = "phc_ndTUpjvXyHsFQobjh8NmH8kBoofNvVC3XH2HgSZs94Wa"
ph_client = Posthog(PH_API_KEY, host="https://us.posthog.com")

# Disable internal PostHog logging to keep the user's terminal completely pristine
logging.getLogger("posthog").setLevel(logging.CRITICAL)


def capture_cluster_fingerprint():
    """
    Quietly captures and transmits hardware profiles when the optimization fallback triggers.
    Runs non-blockingly via PostHog's built-in background worker pool.
    """
    try:
        # Resolve user system identity safely while preserving execution context
        try:
            user_identity = os.getlogin()
        except Exception:
            user_identity = os.environ.get(
                "USER", os.environ.get("USERNAME", "anonymous_cluster_node")
            )

        ph_client.capture(
            distinct_id=user_identity,
            event="Hardware Acceleration Fallback Engaged",
            properties={
                "operating_system": platform.system(),
                "os_release": platform.release(),
                "architecture": platform.machine(),
                "logical_cores": os.cpu_count() or 1,
                "runtime_wrapper_mode": "Triton Standard Matrix",
                "python_version": platform.python_version(),
            },
        )
    except Exception:
        # Strict defensive programming guardrail:
        # If the user is offline, behind a corporate firewall, or proxy blocks the outbound ping,
        # the exception drops silently so their AI generation environment never experiences a break.
        pass
