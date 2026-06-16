Set-Content src/renorm/auth.py -Value @'
import os
import base64
import hmac
import hashlib
import json
import time

def check_enterprise_license():
    token = os.getenv("RENORM_ENTERPRISE_KEY")
    if not token:
        raise PermissionError("🛑 Security Error: Valid RENORM_ENTERPRISE_KEY environment variable not found.")
    try:
        payload_b64, signature = token.split(".")
        verification_secret = "PutAVeryLongRandomStringOfCharactersHere2026!"
        expected_signature = hmac.new(
            verification_secret.encode('utf-8'),
            payload_b64.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            raise PermissionError("🛑 Security Alert: Enterprise license token signature verification failed.")
        payload_json = base64.b64decode(payload_b64.encode('utf-8')).decode('utf-8')
        claims = json.loads(payload_json)
        if time.time() > claims["expiry"]:
            raise TimeoutError("🛑 Trial Lease Expired: Please contact sales to renew your cluster allotment.")
        return claims
    except (ValueError, KeyError):
        raise ValueError("🛑 Core License Error: License token structure is corrupted or unreadable.")
'@