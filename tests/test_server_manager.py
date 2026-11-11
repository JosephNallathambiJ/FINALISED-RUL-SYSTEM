"""Tests for server_manager: lifecycle, network detection, access URLs, and client registry."""

import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.server_manager import ServerManager, detect_vpn_and_tailscale, is_port_bound


def test_detect_vpn_and_tailscale():
    """Verify VPN interface inspection returns valid schema."""
    vpn_info = detect_vpn_and_tailscale()
    assert "tailscale_installed" in vpn_info
    assert "vpn_active" in vpn_info
    assert "vpn_interfaces" in vpn_info
    assert isinstance(vpn_info["vpn_interfaces"], list)


def test_server_manager_initial_state():
    """Verify ServerManager attributes and URL generation."""
    sm = ServerManager()
    assert sm.port == 8080
    assert sm.protocol == "http"

    urls = sm.get_access_urls()
    assert "local_url" in urls
    assert "lan_url" in urls
    assert "http://127.0.0.1:8080" in urls["local_url"]


def test_server_manager_client_registry():
    """Verify tracking and disconnecting remote client sessions."""
    sm = ServerManager()
    client_id = "test_cli_123"

    sm.register_client_heartbeat(
        client_id=client_id,
        ip_address="192.168.1.150",
        device_name="Secondary Monitoring Node",
        role="MONITORING",
    )

    status = sm.get_status()
    assert status["connected_clients_count"] >= 1

    clients = status["connected_clients"]
    matched = [c for c in clients if c["client_id"] == client_id]
    assert len(matched) == 1
    assert matched[0]["device_name"] == "Secondary Monitoring Node"
    assert matched[0]["ip_address"] == "192.168.1.150"

    # Disconnect client
    assert sm.disconnect_client(client_id) is True
    status2 = sm.get_status()
    assert len([c for c in status2["connected_clients"] if c["client_id"] == client_id]) == 0


def test_server_manager_test_connection_unreachable():
    """Verify test_connection gracefully reports unreachable targets."""
    sm = ServerManager()
    res = sm.test_connection("http://192.0.2.1:65432")
    assert res["reachable"] is False
    assert "error" in res

