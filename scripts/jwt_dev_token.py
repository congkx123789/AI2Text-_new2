#!/usr/bin/env python3
"""
Generate a development JWT token for API authentication.

Usage:
    python scripts/jwt_dev_token.py
    
The token is valid for 24 hours and uses HS256 algorithm with "dev" secret.
For production, switch to RS256 with proper key management.
"""

import time
import sys

try:
    import jwt
except ImportError:
    print("Error: PyJWT is not installed.", file=sys.stderr)
    print("Install it with: pip install PyJWT", file=sys.stderr)
    sys.exit(1)

def generate_dev_token(expiry_hours=24):
    """Generate a JWT token for development."""
    now = int(time.time())
    
    payload = {
        "sub": "dev-user",
        "iat": now,
        "exp": now + (expiry_hours * 3600),
        "aud": "ai2text-asr",
        "iss": "ai2text-dev"
    }
    
    token = jwt.encode(payload, "dev", algorithm="HS256")
    return token

if __name__ == "__main__":
    token = generate_dev_token()
    print(token)
    
    # Also print usage instructions to stderr so token alone goes to stdout
    print("\n--- Usage ---", file=sys.stderr)
    print(f"Authorization: Bearer {token}", file=sys.stderr)
    print("\nOr export as environment variable:", file=sys.stderr)
    print(f"export AUTH_TOKEN='{token}'", file=sys.stderr)

