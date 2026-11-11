"""Cryptographic device pairing manager for two-computer distributed predictive maintenance."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import secrets
import string
import time
from typing import Any, Dict, List, Optional, Tuple

PAIRED_DEVICES_FILE = Path(__file__).resolve().parents[1] / "paired_devices.json"


class PairingCode:
    """Represents a temporary, single-use, expiring pairing code."""

    def __init__(self, code: str, ttl_seconds: int = 600, max_uses: int = 1):
        self.code = code
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl_seconds
        self.max_uses = max_uses
        self.use_count = 0

    @property
    def is_valid(self) -> bool:
        return time.time() < self.expires_at and self.use_count < self.max_uses

    @property
    def remaining_seconds(self) -> int:
        return max(0, int(self.expires_at - time.time()))


class DevicePairingManager:
    """Coordinates secure mutual device enrollment between Monitoring and Prediction machines."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or PAIRED_DEVICES_FILE
        self.active_codes: Dict[str, PairingCode] = {}
        self.paired_devices: Dict[str, Dict[str, Any]] = self._load_paired_devices()

    def _load_paired_devices(self) -> Dict[str, Dict[str, Any]]:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                print(f"[PairingManager] Warning: Failed to load {self.storage_path}: {e}")
        return {}

    def _save_paired_devices(self) -> None:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.paired_devices, f, indent=2)
        except Exception as e:
            print(f"[PairingManager] Warning: Failed to save paired devices: {e}")

    def generate_pairing_code(self, ttl_seconds: int = 600) -> Dict[str, Any]:
        """Generate a cryptographically secure, uppercase, alphanumeric single-use code.
        
        Format: PAIR-XXXXXX
        """
        # Purge expired codes first
        self.prune_expired_codes()

        alphabet = string.ascii_uppercase + string.digits
        # Avoid visually ambiguous characters: 0, O, 1, I
        cleaned_alphabet = [c for c in alphabet if c not in ("0", "O", "1", "I")]
        random_suffix = "".join(secrets.choice(cleaned_alphabet) for _ in range(6))
        code_str = f"PAIR-{random_suffix}"

        pairing_obj = PairingCode(code=code_str, ttl_seconds=ttl_seconds, max_uses=1)
        self.active_codes[code_str] = pairing_obj

        # Do NOT log the plain code in logs per security requirements
        return {
            "status": "SUCCESS",
            "pairing_code": code_str,
            "expires_in_seconds": ttl_seconds,
            "expires_at_iso": datetime.datetime.fromtimestamp(
                pairing_obj.expires_at, datetime.timezone.utc
            ).isoformat(),
        }

    def prune_expired_codes(self) -> None:
        """Discard expired or already-used pairing codes."""
        now = time.time()
        to_del = [c for c, obj in self.active_codes.items() if not obj.is_valid]
        for c in to_del:
            del self.active_codes[c]

    def exchange_code_for_token(
        self,
        pairing_code: str,
        device_name: str,
        device_role: str,
        remote_ip: str,
        hardware_model: Optional[str] = None,
        os_info: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate single-use pairing code and exchange for persistent device token.
        
        Returns (success, message, token_payload).
        """
        self.prune_expired_codes()
        code_clean = pairing_code.strip().upper()

        if code_clean not in self.active_codes:
            return False, "Invalid, expired, or already-used pairing code", None

        obj = self.active_codes[code_clean]
        if not obj.is_valid:
            del self.active_codes[code_clean]
            return False, "Pairing code has expired", None

        # Mark code as used immediately (single-use defense)
        obj.use_count += 1
        del self.active_codes[code_clean]

        # Generate persistent credentials
        device_id = f"dev_{secrets.token_hex(6)}"
        device_token = secrets.token_urlsafe(32)
        registered_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        record = {
            "device_id": device_id,
            "device_name": device_name or "Remote Host",
            "device_role": device_role.upper() if device_role else "MONITORING",
            "ip_address": remote_ip,
            "hardware_model": hardware_model or "Standard CPU Host",
            "os_info": os_info or "Unknown OS",
            "registered_at": registered_at,
            "last_seen": registered_at,
            "status": "ACTIVE",
            "token_hash": secrets.token_hex(16),  # internal marker
        }

        # Store in persistent paired devices
        self.paired_devices[device_id] = record
        self._save_paired_devices()

        # Register in active security manager device tokens
        from control.authentication import security_mgr
        security_mgr.device_tokens[device_id] = {
            "token": device_token,
            "role": record["device_role"],
            "device_name": record["device_name"],
        }

        return True, "Device paired successfully", {
            "device_id": device_id,
            "device_token": device_token,
            "device_role": record["device_role"],
            "registered_at": registered_at,
        }

    def verify_device(self, device_id: str, token: str) -> bool:
        """Check whether a device is registered, active, and token matches."""
        if device_id not in self.paired_devices:
            return False
        dev = self.paired_devices[device_id]
        if dev.get("status") != "ACTIVE":
            return False
        from control.authentication import security_mgr
        entry = security_mgr.device_tokens.get(device_id)
        if entry and secrets.compare_digest(entry["token"], token):
            return True
        return False

    def update_heartbeat(self, device_id: str, ip_address: Optional[str] = None) -> None:
        """Update last communication timestamp for a connected device."""
        if device_id in self.paired_devices:
            self.paired_devices[device_id]["last_seen"] = (
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            )
            if ip_address:
                self.paired_devices[device_id]["ip_address"] = ip_address
            self._save_paired_devices()

    def list_devices(self) -> List[Dict[str, Any]]:
        """Return all registered devices without leaking device tokens."""
        devices = []
        for dev_id, data in self.paired_devices.items():
            entry = dict(data)
            entry.pop("token_hash", None)
            devices.append(entry)
        return devices

    def revoke_device(self, device_id: str) -> bool:
        """Revoke authorization for a device."""
        if device_id in self.paired_devices:
            self.paired_devices[device_id]["status"] = "REVOKED"
            self._save_paired_devices()
            from control.authentication import security_mgr
            if device_id in security_mgr.device_tokens:
                del security_mgr.device_tokens[device_id]
            return True
        return False

    def delete_device(self, device_id: str) -> bool:
        """Completely remove a device from paired registry."""
        if device_id in self.paired_devices:
            del self.paired_devices[device_id]
            self._save_paired_devices()
            from control.authentication import security_mgr
            if device_id in security_mgr.device_tokens:
                del security_mgr.device_tokens[device_id]
            return True
        return False


pairing_mgr = DevicePairingManager()

