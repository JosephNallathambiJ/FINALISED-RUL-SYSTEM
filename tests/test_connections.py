"""Tests for network connection, device interfaces, and peer pairing."""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.device_manager import DeviceManager, get_network_interfaces, get_system_hardware_info
from control.network_manager import NetworkManager


def test_network_interfaces_detection():
    """Verify system network interfaces can be enumerated."""
    interfaces = get_network_interfaces()
    assert isinstance(interfaces, dict)
    # Loopback or local interface should be found
    assert len(interfaces) > 0


def test_system_hardware_info():
    """Verify hardware profile extraction."""
    info = get_system_hardware_info()
    assert "os_name" in info
    assert "cpu_logical_cores" in info
    assert info["cpu_logical_cores"] >= 1
    assert "ram_total_gb" in info
    assert info["ram_total_gb"] > 0.0


def test_network_manager_unreachable_ping():
    """Verify ping_remote safely fails when contacting an unused/unreachable IP."""
    net_mgr = NetworkManager()
    # 192.0.2.1 is TEST-NET-1 (RFC 5737 - reserved non-routable)
    res = net_mgr.ping_remote(host="192.0.2.1", port=65432, timeout_sec=0.5)
    assert res["status"] == "OFFLINE"
    assert res["reachable"] is False


def test_network_manager_peer_pairing():
    """Verify peer connection and disconnect state changes."""
    net_mgr = NetworkManager()
    assert not net_mgr.is_connected

    # Unpaired state
    state = net_mgr.get_connection_status()
    assert state["status"] == "DISCONNECTED"

    # Pair with target
    net_mgr.disconnect_peer()
    assert not net_mgr.is_connected

