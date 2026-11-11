"""Device identification, local network discovery, and configuration persistence."""

from __future__ import annotations

import json
import os
import platform
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

CONFIG_FILE = Path(__file__).resolve().parents[1] / "control_config.json"


def get_primary_ip() -> str:
    """Determine the primary local IPv4 address connected to the local network."""
    # Method 1: Connect a UDP socket (does not transmit data) to resolve routing IP
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    # Method 2: Inspect network interfaces via psutil
    try:
        addrs = psutil.net_if_addrs()
        candidates = []
        for iface_name, iface_addrs in addrs.items():
            for addr in iface_addrs:
                if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                    # Prefer Wi-Fi or Ethernet interfaces
                    lower_name = iface_name.lower()
                    if any(prefix in lower_name for prefix in ["en", "eth", "wlan", "wi-fi", "wl"]):
                        return addr.address
                    candidates.append(addr.address)
        if candidates:
            return candidates[0]
    except Exception:
        pass

    # Fallback to localhost
    return "127.0.0.1"


def get_network_interfaces() -> Dict[str, Dict[str, Any]]:
    """List all available physical and virtual network interfaces with their IPv4 addresses."""
    interfaces: Dict[str, Dict[str, Any]] = {}
    try:
        addrs = psutil.net_if_addrs()
        stats = psutil.net_if_stats()
        for name, addr_list in addrs.items():
            ipv4_list = [a.address for a in addr_list if a.family == socket.AF_INET]
            if ipv4_list:
                is_up = stats[name].isup if name in stats else True
                speed = stats[name].speed if name in stats else 0
                interfaces[name] = {
                    "name": name,
                    "ip": ipv4_list[0],
                    "all_ips": ipv4_list,
                    "is_up": is_up,
                    "speed_mbps": speed,
                }
    except Exception:
        pass
    return interfaces


def get_system_hardware_info() -> Dict[str, Any]:
    """Inspect and return platform, OS, CPU architecture, and core counts."""
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "os_name": platform.system(),
        "machine": platform.machine(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or platform.machine() or "Unknown CPU",
        "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        "cpu_physical_cores": psutil.cpu_count(logical=False) or 1,
        "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
        "cpu_logical_cores": psutil.cpu_count(logical=True) or 1,
        "python_version": platform.python_version(),
    }
    try:
        freq = psutil.cpu_freq()
        if freq:
            info["cpu_freq_current_mhz"] = round(freq.current, 1)
            info["cpu_freq_max_mhz"] = round(freq.max, 1) if freq.max else None
            info["cpu_freq_min_mhz"] = round(freq.min, 1) if freq.min else None
    except Exception:
        pass

    try:
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
    except Exception:
        pass

    return info


class DeviceManager:
    """Manages local identity, network pairing state, and configuration persistence."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "role": "local_demo",  # "monitoring", "prediction", "local_demo"
        "device_name": socket.gethostname(),
        "monitoring_port": 8001,
        "prediction_port": 8000,
        "dashboard_port": 8080,
        "remote_ip": "",
        "remote_port": 8000,
        "sampling_interval": 5.0,
        "api_key": "",
        "profile_name": "desktop",
        "auto_sync": True,
        "auto_start_monitoring": True,
    }

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_FILE
        self.config: Dict[str, Any] = self.load_config()

    @property
    def role(self) -> str:
        return self.config.get("role", "LOCAL_DEMO").upper()

    def set_role(self, role: str) -> None:
        self.config["role"] = role.upper()
        self.save_config()

    @property
    def primary_ip(self) -> str:
        return get_primary_ip()

    @property
    def device_id(self) -> str:
        return self.config.get("device_id", self.config.get("device_name", socket.gethostname()))

    def get_device_info(self) -> Dict[str, Any]:
        hw = get_system_hardware_info()
        return {
            "device_id": self.device_id,
            "role": self.role,
            "primary_ip": self.primary_ip,
            "hostname": hw["hostname"],
            "os_name": hw["os_name"],
            "machine": hw["machine"],
            "processor": hw["processor"],
            "python_version": hw["python_version"],
            "cpu_physical_cores": hw["cpu_physical_cores"],
            "cpu_logical_cores": hw["cpu_logical_cores"],
            "ram_total_gb": hw.get("ram_total_gb", 0.0),
        }

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON or initialize with defaults."""
        config = dict(self.DEFAULT_CONFIG)
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    if isinstance(saved, dict):
                        config.update(saved)
            except Exception:
                pass
        return config

    def save_config(self) -> None:
        """Persist current configuration to disk."""
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save config to {self.config_path}: {e}")

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update and persist configuration attributes."""
        self.config.update(updates)
        self.save_config()
        return self.config

    def get_device_summary(self) -> Dict[str, Any]:
        """Compile a complete status summary of the local device and network state."""
        primary_ip = get_primary_ip()
        hw = get_system_hardware_info()
        role = self.config.get("role", "local_demo")

        # Determine appropriate API port for current role
        if role == "monitoring":
            active_api_port = self.config.get("monitoring_port", 8001)
        elif role == "prediction":
            active_api_port = self.config.get("prediction_port", 8000)
        else:
            active_api_port = self.config.get("dashboard_port", 8080)

        return {
            "hostname": hw["hostname"],
            "primary_ip": primary_ip,
            "interfaces": get_network_interfaces(),
            "role": role,
            "active_api_port": active_api_port,
            "dashboard_port": self.config.get("dashboard_port", 8080),
            "monitoring_port": self.config.get("monitoring_port", 8001),
            "prediction_port": self.config.get("prediction_port", 8000),
            "remote_ip": self.config.get("remote_ip", ""),
            "remote_port": self.config.get("remote_port", 8000),
            "sampling_interval": self.config.get("sampling_interval", 5.0),
            "profile_name": self.config.get("profile_name", "desktop"),
            "hardware": hw,
        }
