"""Integration tests for FastAPI Prognostics and Telemetry Service."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from starlette.testclient import TestClient

PRED_DIR = Path(__file__).resolve().parents[1] / "iqac-prediction"
sys.path.insert(0, str(PRED_DIR))

from api import app
from database import init_db, insert_daily_record, insert_telemetry

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_dashboard_and_health_endpoints():
    """Verify HTML dashboard and health endpoints respond with HTTP 200."""
    res_dash = client.get("/")
    assert res_dash.status_code == 200
    assert "IQAC Prognostics & RUL Engine" in res_dash.text

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "healthy"
    assert "physics" in data["supported_modes"]


def test_components_profiles():
    """Verify component profiles listing."""
    res = client.get("/components/profiles")
    assert res.status_code == 200
    data = res.json()
    assert "desktop" in data
    assert "server" in data
    assert "laptop" in data


def test_telemetry_post_and_metrics_get():
    """Test POST /telemetry ingestion and GET /devices/{device_id}/metrics aggregation."""
    import uuid
    dev_id = f"test_api_{uuid.uuid4().hex[:8]}"
    payload = {
        "device_id": dev_id,
        "cpu_temp": 52.5,
        "fan_rpm": 2500,
        "cpu_usage": 45.0,
        "voltage_v": 1.18,
        "is_spike": False,
    }
    post_res = client.post("/telemetry", json=payload)
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "success"

    metrics_res = client.get(f"/devices/{dev_id}/metrics")
    assert metrics_res.status_code == 200
    metrics = metrics_res.json()
    assert metrics["device_id"] == dev_id
    assert metrics["avg_temp"] == 52.5
    assert metrics["cpu_usage_avg"] == 45.0


def test_structured_rul_api_response():
    """Verify GET /devices/{device_id}/rul complies with Section 4 structured schema."""
    import uuid
    dev_id = f"test_rul_schema_{uuid.uuid4().hex[:8]}"
    # Seed telemetry
    for i in range(10):
        client.post("/telemetry", json={
            "device_id": dev_id,
            "cpu_temp": 50.0 + i,
            "fan_rpm": 2000,
            "cpu_usage": 30.0,
            "voltage_v": 1.15,
        })

    rul_res = client.get(f"/devices/{dev_id}/rul?mode=physics&profile=desktop")
    assert rul_res.status_code == 200
    data = rul_res.json()

    # Required Section 4 schema fields
    assert data["component"] == "CPU"
    assert "rul_hours" in data
    assert "rul_days" in data
    assert "health_percentage" in data
    assert "degradation_percentage" in data
    assert "risk_level" in data
    assert data["prediction_method"] == "PHYSICS_BASED"
    assert "confidence_interval" in data
    assert "data_quality" in data
    assert "model_version" in data
    assert "limitations" in data
    assert isinstance(data["limitations"], list)

    # Backward compatibility fields
    assert "estimated_rul_hours" in data
    assert "health_score_pct" in data


def test_history_and_diagnose_endpoints():
    """Verify GET /devices/{id}/history and GET /devices/{id}/diagnose."""
    dev_id = "test_hist_device"
    insert_daily_record(dev_id, {
        "day": "2026-06-15",
        "reading_count": 40,
        "average_temperature_c": 55.0,
        "average_cpu_usage": 32.0,
        "spike_count": 1,
        "updated_at": "2026-06-15T23:59:59+00:00",
    })

    hist_res = client.get(f"/devices/{dev_id}/history")
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert "2026-06-15" in hist["days"]
    assert len(hist["cumulative_damages"]) == 1

    from unittest.mock import patch
    with patch("agent.detect_ollama_model", return_value=None):
        diag_res = client.get(f"/devices/{dev_id}/diagnose")
        assert diag_res.status_code == 200
        diag = diag_res.json()
        assert "diagnostic_report" in diag
        assert "SECTION 1: GROUND-TRUTH TELEMETRY" in diag["diagnostic_report"]
