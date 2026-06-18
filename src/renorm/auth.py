import base64
import hashlib
import hmac
import json
import os
import time


def check_enterprise_license():
    """Validate the enterprise license token stored in
    the RENORM_ENTERPRISE_KEY environment variable.

    Expected token format:
        base64_payload.hex_hmac_signature
    """
    token = os.getenv("RENORM_ENTERPRISE_KEY")
    if not token:
        raise PermissionError("Valid RENORM_ENTERPRISE_KEY environment variable not found.")

    try:
        payload_b64, signature = token.split(".", maxsplit=1)
    except ValueError as exc:
        raise ValueError("Invalid license token format.") from exc

    verification_secret = "PutAVeryLongRandomStringOfCharactersHere2026!"

    expected_signature = hmac.new(
        verification_secret.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected_signature):
        raise PermissionError("License signature verification failed.")

    try:
        payload_json = base64.b64decode(payload_b64).decode("utf-8")
        claims = json.loads(payload_json)
    except Exception as exc:
        raise ValueError("License payload is invalid.") from exc

    expiry = claims.get("expiry")
    if expiry is None:
        raise ValueError("License payload missing expiry field.")

    if time.time() > float(expiry):
        raise TimeoutError("License has expired.")

    return claims
