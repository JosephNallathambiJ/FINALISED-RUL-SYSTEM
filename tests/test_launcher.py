"""Tests for overall_run launcher utilities, role selection, and port allocation."""

import os
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.device_manager import DeviceManager, get_primary_ip
from overall_run import check_dependencies, find_free_port


def test_dependency_check():
    """Verify check_dependencies returns True when required packages are installed."""
    assert check_dependencies() is True


def test_find_free_port():
    """Verify find_free_port locates an open TCP port."""
    port = find_free_port(8080)
    assert 8080 <= port < 8150


def test_device_manager_initialization(tmp_path):
    """Verify DeviceManager creates default configuration and persists changes."""
    cfg_path = tmp_path / "test_config.json"
    mgr = DeviceManager(config_path=cfg_path)
    
    assert mgr.role in ["MONITORING", "PREDICTION", "LOCAL_DEMO"]
    assert mgr.primary_ip is not None
    assert len(mgr.device_id) > 0

    # Test role update and persistence
    mgr.set_role("PREDICTION")
    assert mgr.role == "PREDICTION"

    # Reload from file to verify persistence
    mgr2 = DeviceManager(config_path=cfg_path)
    assert mgr2.role == "PREDICTION"


def test_primary_ip_detection():
    """Verify get_primary_ip returns valid IPv4 address."""
    ip = get_primary_ip()
    assert isinstance(ip, str)
    parts = ip.split(".")
    assert len(parts) == 4
    for p in parts:
        assert p.isdigit()
        assert 0 <= int(p) <= 255

