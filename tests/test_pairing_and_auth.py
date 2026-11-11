"""Tests for cryptographic pairing, single-use codes, token exchange, and security controls."""

import time
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.authentication import RateLimiter, SecurityManager, hash_secret, verify_secret
from control.pairing_manager import DevicePairingManager


def test_hash_and_verify_secret():
    """Verify PBKDF2-HMAC-SHA256 password hashing and constant-time verification."""
    password = "SuperSecureAdminPassword123!"
    salt, hash_val = hash_secret(password)
    
    assert len(salt) > 0
    assert len(hash_val) > 0
    assert verify_secret(password, salt, hash_val) is True
    assert verify_secret("WrongPassword", salt, hash_val) is False


def test_rate_limiter():
    """Verify sliding-window rate limiter blocks excessive requests."""
    limiter = RateLimiter(requests_per_minute=3)
    ip = "192.168.1.99"
    assert limiter.is_allowed(ip) is True
    assert limiter.is_allowed(ip) is True
    assert limiter.is_allowed(ip) is True
    # 4th request within window must be rejected
    assert limiter.is_allowed(ip) is False


def test_pairing_code_generation_and_single_use(tmp_path):
    """Verify pairing codes are single-use, formatted correctly, and expire."""
    store_file = tmp_path / "test_paired.json"
    mgr = DevicePairingManager(storage_path=store_file)

    gen_res = mgr.generate_pairing_code(ttl_seconds=300)
    assert gen_res["status"] == "SUCCESS"
    code = gen_res["pairing_code"]
    assert code.startswith("PAIR-")
    assert len(code) == 11  # PAIR- + 6 chars

    # First exchange succeeds
    ok, msg, creds = mgr.exchange_code_for_token(
        pairing_code=code,
        device_name="Test-Monitor-Node",
        device_role="MONITORING",
        remote_ip="192.168.1.105",
    )
    assert ok is True
    assert creds is not None
    assert "device_token" in creds
    device_id = creds["device_id"]

    # Second exchange with the SAME code must fail (single-use constraint)
    ok2, msg2, creds2 = mgr.exchange_code_for_token(
        pairing_code=code,
        device_name="Impostor-Node",
        device_role="MONITORING",
        remote_ip="192.168.1.106",
    )
    assert ok2 is False
    assert creds2 is None

    # Verify device exists in paired list
    devices = mgr.list_devices()
    matched = [d for d in devices if d["device_id"] == device_id]
    assert len(matched) == 1
    assert matched[0]["status"] == "ACTIVE"

    # Revoke device
    assert mgr.revoke_device(device_id) is True
    assert mgr.verify_device(device_id, creds["device_token"]) is False


def test_pairing_code_expiration(tmp_path):
    """Verify expired pairing codes cannot be exchanged."""
    store_file = tmp_path / "test_paired_exp.json"
    mgr = DevicePairingManager(storage_path=store_file)

    # 1 second TTL
    gen_res = mgr.generate_pairing_code(ttl_seconds=1)
    code = gen_res["pairing_code"]
    time.sleep(1.2)

    ok, msg, _ = mgr.exchange_code_for_token(
        pairing_code=code,
        device_name="Late-Node",
        device_role="MONITORING",
        remote_ip="192.168.1.105",
    )
    assert ok is False
    assert "expired" in msg.lower()

