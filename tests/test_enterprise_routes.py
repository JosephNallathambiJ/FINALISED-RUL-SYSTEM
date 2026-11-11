"""Tests for enterprise endpoints: Server Hosting, Device Pairing, Hardware Inventory, and Series."""

import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.app import app


def test_server_management_endpoints():
    """Verify server status and reachability test routes."""
    with TestClient(app) as client:
        res = client.get("/api/server/status")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "access_urls" in data
        assert "connected_clients" in data
        assert "vpn_info" in data

        # Test reachability route
        test_res = client.post("/api/server/test", json={"url": "http://127.0.0.1:8080/health"})
        assert test_res.status_code == 200
        assert "reachable" in test_res.json()

        # Clients route
        cli_res = client.get("/api/server/clients")
        assert cli_res.status_code == 200
        assert isinstance(cli_res.json(), list)


def test_pairing_workflow_endpoints():
    """Verify pairing code generation and exchange via API."""
    with TestClient(app) as client:
        # 1. Generate single-use pairing code
        gen_res = client.post("/api/pairing/code/generate", json={"ttl_seconds": 300})
        assert gen_res.status_code == 200
        gen_data = gen_res.json()
        assert "pairing_code" in gen_data
        code = gen_data["pairing_code"]

        # 2. Exchange code for credentials
        ex_res = client.post(
            "/api/pairing/exchange",
            json={
                "pairing_code": code,
                "device_name": "Lab-Monitoring-Rig",
                "device_role": "MONITORING",
                "hardware_model": "Dell PowerEdge",
            },
        )
        assert ex_res.status_code == 200
        ex_data = ex_res.json()
        assert ex_data["status"] == "SUCCESS"
        assert "credentials" in ex_data
        dev_id = ex_data["credentials"]["device_id"]

        # 3. List devices
        dev_res = client.get("/api/pairing/devices")
        assert dev_res.status_code == 200
        devices = dev_res.json()
        matched = [d for d in devices if d["device_id"] == dev_id]
        assert len(matched) == 1

        # 4. Revoke device
        rev_res = client.post("/api/pairing/devices/revoke", json={"device_id": dev_id})
        assert rev_res.status_code == 200
        assert rev_res.json()["status"] == "REVOKED"


def test_hardware_inventory_and_monitoring_series():
    """Verify hardware inventory, filtered time-series ranges, and recommendations."""
    with TestClient(app) as client:
        # Hardware inventory
        inv_res = client.get("/api/inventory/hardware")
        assert inv_res.status_code == 200
        inv_data = inv_res.json()
        assert "os_name" in inv_data
        assert "physical_cores" in inv_data
        assert "ram_total_gb" in inv_data

        # Time series range (5m)
        series_res = client.get("/api/monitoring/series?range_window=5m")
        assert series_res.status_code == 200
        series_data = series_res.json()
        assert "temperatures" in series_data
        assert "utilizations" in series_data
        assert "timestamps" in series_data

        # Thermal spikes list
        spikes_res = client.get("/api/thermal/spikes")
        assert spikes_res.status_code == 200
        assert isinstance(spikes_res.json(), list)

        # Maintenance recommendations
        recs_res = client.get("/api/agent/recommendations")
        assert recs_res.status_code == 200
        recs = recs_res.json()
        assert isinstance(recs, list)
        assert len(recs) >= 1
        assert "title" in recs[0]
        assert "priority" in recs[0]

