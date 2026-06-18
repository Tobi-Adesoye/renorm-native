import base64
import hashlib
import hmac
import json
import os
import sys
import time

# Self-bootstrapping runtime path correction
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class RenormEnvironmentManager:
    """
    Manages host OS allocations, overrides driver variables, and enforces
    time-expiring cryptographic corporate token leases completely offline.
    """

    def __init__(self, secret_signing_key: str = "RENORM_CORE_SECRET_2026"):
        self.secret_key = secret_signing_key.encode("utf-8")
        self.license_status = "PENDING_VERIFICATION"
        self.corporate_id = "COMMUNITY_EDITION"
        self.max_nodes = 1

        self._validate_enterprise_token()
        self._apply_hardware_shields()

    def _validate_enterprise_token(self) -> None:
        token = os.environ.get("RENORM_ENTERPRISE_KEY")

        if not token:
            print(
                "[Renorm-Gateway] WARNING: No Enterprise License Key found. "
                "Operating in Throttled Community Mode."
            )
            self.license_status = "THROTTLED_COMMUNITY"
            return

        try:
            if "." not in token:
                raise ValueError("Malformed token structure")

            payload_b64, signature_hex = token.split(".")
            expected_sig = hmac.new(
                self.secret_key,
                payload_b64.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

            if not hmac.compare_digest(expected_sig, signature_hex):
                print("[Renorm-Gateway] CRITICAL: Invalid signature")
                self.license_status = "INVALID_SIGNATURE"
                return

            payload_json = base64.b64decode(payload_b64).decode("utf-8")
            claims = json.loads(payload_json)

            if time.time() > claims.get("expiry", 0):
                print("[Renorm-Gateway] CRITICAL: License expired")
                self.license_status = "LEASE_EXPIRED"
                return

            self.corporate_id = claims.get("corporate_id", "UNKNOWN")
            self.max_nodes = claims.get("max_nodes", 1)
            self.license_status = "ENTERPRISE_UNLOCKED"

            print(f"[Renorm-Gateway] SUCCESS: Authenticated [{self.corporate_id}]")

        except Exception as e:
            print(f"[Renorm-Gateway] ERROR: {e}")
            self.license_status = "CORRUPTED_TOKEN"

    def _apply_hardware_shields(self) -> None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        if device == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
        else:
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

    def is_acceleration_allowed(self) -> bool:
        return self.license_status == "ENTERPRISE_UNLOCKED"
