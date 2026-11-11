"""Tests for background process supervisor, stress tester, and dashboard API routes."""

import socket
import sys
import time
from pathlib import Path
import pytest
from starlette.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.process_manager import LogBuffer, ProcessManager
from control.stress_tester import ControlledStressTester
from dashboard.app import app


def test_log_buffer_rotation_and_filtering():
    """Verify thread-safe LogBuffer caps at max capacity and filters by severity."""
    buf = LogBuffer(max_lines=50)
    for i in range(70):
        level = "ERROR" if i % 10 == 0 else "INFO"
        buf.append("TestModule", f"Message {i}", level=level)

    logs = buf.get_logs(limit=100)
    assert len(logs) == 50  # Enforced max capacity
    assert logs[-1]["message"] == "Message 69"

    # Test severity filtering
    error_logs = buf.get_logs(level="ERROR")
    for l in error_logs:
        assert l["level"] == "ERROR"


def test_process_manager_port_detection():
    """Verify ProcessManager detects active listening TCP ports."""
    pm = ProcessManager()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        bound_port = s.getsockname()[1]
        assert pm.is_port_in_use(bound_port) is True

    # After close, port should no longer be in use
    assert pm.is_port_in_use(bound_port) is False


def test_controlled_stress_tester_lifecycle():
    """Verify ControlledStressTester starts bounded load and terminates cleanly."""
    tester = ControlledStressTester()
    res = tester.start(target_load_pct=30, max_duration_sec=3, temp_cutoff_c=99.0)
    assert res["status"] == "STARTED"
    assert tester.is_running

    time.sleep(0.5)
    status = tester.get_status()
    assert status["active"] is True

    stop_res = tester.stop()
    assert stop_res["status"] == "STOPPED"
    assert not tester.is_running


def test_dashboard_api_endpoints():
    """Verify dashboard REST API endpoints operate as expected."""
    with TestClient(app) as client:
        # 1. Health check
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "HEALTHY"

        # 2. System Info
        res = client.get("/api/system/info")
        assert res.status_code == 200
        info = res.json()
        assert "role" in info
        assert "primary_ip" in info
        assert "hostname" in info

        # 3. Measurement Controls
        res = client.post("/api/measurements/pause")
        assert res.status_code == 200
        assert res.json()["is_paused"] is True

        res = client.post("/api/measurements/resume")
        assert res.status_code == 200
        assert res.json()["is_paused"] is False

        res = client.post("/api/measurements/interval", json={"interval": 2.5})
        assert res.status_code == 200
        assert res.json()["interval"] == 2.5

        # 4. RUL Calculation (Physics Mode)
        res = client.post(
            "/api/rul/calculate",
            json={
                "mode": "physics",
                "component_profile": "DESKTOP_HIGH_END",
                "avg_temp": 55.0,
                "operating_hours_per_day": 16.0,
            },
        )
        assert res.status_code == 200
        rul_data = res.json()
        assert rul_data["rul_hours"] > 0
        assert rul_data["health_percentage"] > 0
        assert rul_data["risk_level"] in ["LOW", "NORMAL", "MODERATE", "HIGH", "CRITICAL"]

        # 5. RUL Calculation (Hybrid Mode)
        res = client.post(
            "/api/rul/calculate",
            json={
                "mode": "hybrid",
                "component_profile": "SERVER_XEON_EPYC",
                "avg_temp": 50.0,
            },
        )
        assert res.status_code == 200
        assert "HYBRID" in res.json()["prediction_method"].upper()

        # 6. Stress Status
        res = client.get("/api/stress/status")
        assert res.status_code == 200

        # 7. Logs Endpoint
        res = client.get("/api/logs?limit=10")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

        # 8. Settings Update
        res = client.post("/api/settings", json={"sampling_interval": 5.0})
        assert res.status_code == 200
        assert res.json()["status"] == "SUCCESS"
