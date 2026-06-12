"""
Renorm-Native: Enterprise Gateway & License Referee
Coordinates machine allocation variables and validates corporate cryptographic leases.
"""

import os
import sys
import time
import base64
import hmac
import hashlib
import json
from typing import Dict, Any, Optional

class RenormEnvironmentManager:
    """
    Manages host OS allocations, overrides driver variables, and enforces
    time-expiring cryptographic corporate token leases completely offline.
    """
    def __init__(self, secret_signing_key: str = "RENORM_CORE_SECRET_2026"):
        self.secret_key = secret_signing_key.encode('utf-8')
        self.license_status = "PENDING_VERIFICATION"
        self.corporate_id = "COMMUNITY_EDITION"
        self.max_nodes = 1
        
        self._validate_enterprise_token()
        self._apply_hardware_shields()

    def _validate_enterprise_token(self) -> None:
        token = os.environ.get("RENORM_ENTERPRISE_KEY", None)
        if not token:
            print("[Renorm-Gateway] WARNING: No Enterprise License Key found. Operating in Throttled Community Mode.")
            self.license_status = "THROTTLED_COMMUNITY"
            return

        try:
            if "." not in token:
                raise ValueError("Malformed token string structure.")
            payload_b64, signature_hex = token.split(".")
            expected_sig = hmac.new(self.secret_key, payload_b64.encode('utf-8'), hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(expected_sig, signature_hex):
                print("[Renorm-Gateway] CRITICAL: Cryptographic signature mismatch! Access Denied.")
                self.license_status = "INVALID_SIGNATURE"
                return
                
            payload_json = base64.b64decode(payload_b64.encode('utf-8')).decode('utf-8')
            claims = json.loads(payload_json)
            
            if time.time() > claims.get("expiry", 0):
                print(f"[Renorm-Gateway] CRITICAL: License lease expired!")
                self.license_status = "LEASE_EXPIRED"
                return
                
            self.corporate_id = claims.get("corporate_id", "UNKNOWN_CORP")
            self.max_nodes = claims.get("max_nodes", 1)
            self.license_status = "ENTERPRISE_UNLOCKED"
            print(f"[Renorm-Gateway] SUCCESS: Authenticated lease for [{self.corporate_id}].")
            
        except Exception as e:
            print(f"[Renorm-Gateway] CRITICAL: Verification engine error: {str(e)}")
            self.license_status = "CORRUPTED_TOKEN"

    def _apply_hardware_shields(self) -> None:
        """
        Dynamically detects backend context and safely applies driver optimization profiles.
        """
        # Safe import check for PyTorch availability
        try:
            import torch
            device_available = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device_available = "cpu"

        # Apply memory virtualization optimizations only if CUDA is present
        if device_available == "cuda":
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
            if sys.platform == "win32":
                os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        else:
            # CPU Fallback Mode structural configurations
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

    def is_acceleration_allowed(self) -> bool:
        return self.license_status == "ENTERPRISE_UNLOCKED"