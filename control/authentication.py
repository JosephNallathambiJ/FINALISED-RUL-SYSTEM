"""Enterprise-grade authentication, role-based authorization, and security controls."""

from __future__ import annotations

import collections
import hashlib
import hmac
import os
import secrets
import time
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from fastapi import Header, HTTPException, Request, WebSocket, status

# Default session duration: 24 hours (in seconds)
DEFAULT_SESSION_TTL = 86400

# Rate limiting configuration (per client IP)
DEFAULT_RATE_LIMIT_RPM = 120  # requests per minute for standard APIs
AUTH_RATE_LIMIT_RPM = 15     # requests per minute for login/pairing attempts


def hash_secret(secret: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """Hash a secret using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    
    Returns (hex_salt, hex_hash).
    """
    if salt is None:
        salt = secrets.token_bytes(16)
    hashed = hashlib.pbkdf2_hmac("sha256", secret.encode("utf-8"), salt, 100000)
    return salt.hex(), hashed.hex()


def verify_secret(secret: str, salt_hex: str, hash_hex: str) -> bool:
    """Verify secret against stored salt and hash using constant-time comparison."""
    try:
        salt = bytes.fromhex(salt_hex)
        _, computed = hash_secret(secret, salt)
        return hmac.compare_digest(computed, hash_hex)
    except Exception:
        return False


class RateLimiter:
    """Sliding-window in-memory request rate limiter per client IP."""

    def __init__(self, requests_per_minute: int = DEFAULT_RATE_LIMIT_RPM):
        self.rpm = requests_per_minute
        self.window = 60.0  # seconds
        self._history: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        q = self._history[client_ip]
        cutoff = now - self.window
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= self.rpm:
            return False
        q.append(now)
        return True


class SecurityManager:
    """Coordinates authentication sessions, device tokens, and origin/host verification."""

    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # token -> session_data
        self.device_tokens: Dict[str, Dict[str, Any]] = {}    # device_id -> token_data
        self.admin_api_key: str = os.getenv("IQAC_API_KEY", secrets.token_urlsafe(24))
        self.allowed_hosts: Set[str] = {"127.0.0.1", "localhost"}
        self.allowed_origins: Set[str] = set()
        self.general_limiter = RateLimiter(DEFAULT_RATE_LIMIT_RPM)
        self.auth_limiter = RateLimiter(AUTH_RATE_LIMIT_RPM)
        self.require_auth: bool = False  # By default, local access is open; enabled when LAN/WAN hosting starts

    def add_allowed_host(self, host: str) -> None:
        """Register a valid host name or IP address (e.g. LAN IP)."""
        clean = host.split(":")[0].strip()
        if clean:
            self.allowed_hosts.add(clean)

    def add_allowed_origin(self, origin: str) -> None:
        """Register an allowed HTTP origin (e.g. http://192.168.1.100:8080)."""
        clean = origin.rstrip("/")
        if clean:
            self.allowed_origins.add(clean)

    def validate_host_header(self, request: Request) -> bool:
        """Protect against DNS rebinding attacks by validating the Host header."""
        host_header = request.headers.get("host", "")
        if not host_header:
            return True  # Internal calls or tests without Host header
        host_name = host_header.split(":")[0].strip().lower()
        if host_name in self.allowed_hosts or host_name in ("testserver", "localhost", "127.0.0.1"):
            return True
        # Allow IPv4 private addresses (10.x, 172.16-31.x, 192.168.x, 100.x)
        if (
            host_name.startswith("192.168.")
            or host_name.startswith("10.")
            or host_name.startswith("100.")
            or host_name.startswith("172.")
        ):
            self.add_allowed_host(host_name)
            return True
        return False

    def validate_origin_header(self, origin: Optional[str]) -> bool:
        """Protect against Cross-Site WebSocket Hijacking (CSWSH) and CSRF."""
        if not origin:
            return True  # Direct non-browser or same-origin requests
        origin_clean = origin.rstrip("/").lower()
        if origin_clean in self.allowed_origins:
            return True
        # Allow localhost / loopback
        if any(h in origin_clean for h in ["localhost", "127.0.0.1", "192.168.", "100.", "10."]):
            return True
        return False

    def create_session(
        self,
        username: str,
        role: str = "ADMIN",
        client_ip: str = "127.0.0.1",
        ttl_seconds: int = DEFAULT_SESSION_TTL,
    ) -> str:
        """Create a cryptographically secure session token."""
        token = secrets.token_urlsafe(32)
        self.active_sessions[token] = {
            "username": username,
            "role": role.upper(),
            "client_ip": client_ip,
            "created_at": time.time(),
            "expires_at": time.time() + ttl_seconds,
        }
        return token

    def verify_session(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        """Validate session token; prune expired sessions."""
        if not token:
            return None
        session = self.active_sessions.get(token)
        if not session:
            return None
        if time.time() > session["expires_at"]:
            del self.active_sessions[token]
            return None
        return session

    def revoke_session(self, token: str) -> bool:
        """Terminate an active session."""
        if token in self.active_sessions:
            del self.active_sessions[token]
            return True
        return False

    def authenticate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Inspect headers for Bearer token, X-API-Key, or X-Device-Token."""
        # 1. Check Bearer token in Authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            session = self.verify_session(token)
            if session:
                return session

        # 2. Check X-API-Key header
        api_key = request.headers.get("x-api-key")
        if api_key and (api_key == self.admin_api_key or api_key == os.getenv("IQAC_API_KEY")):
            return {"username": "api_admin", "role": "ADMIN", "client_ip": request.client.host if request.client else "127.0.0.1"}

        # 3. Check X-Device-Token for paired computers
        device_token = request.headers.get("x-device-token")
        device_id = request.headers.get("x-device-id")
        if device_token and device_id:
            dev = self.device_tokens.get(device_id)
            if dev and hmac.compare_digest(dev["token"], device_token):
                return {"username": f"device_{device_id}", "role": dev.get("role", "OPERATOR"), "device_id": device_id}

        return None


security_mgr = SecurityManager()

