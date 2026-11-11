"""Unit and integration tests for the Agentic AI and Local LLM diagnostic engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT_DIR / "iqac-prediction"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

import agent
from dashboard.app import app

client = TestClient(app)


def test_detect_ollama_model_env_override(monkeypatch):
    """Verify OLLAMA_MODEL environment variable takes precedence."""
    monkeypatch.setenv("OLLAMA_MODEL", "custom-cyber-model:latest")
    assert agent.detect_ollama_model() == "custom-cyber-model:latest"


def test_tool_compute_rul_resilience():
    """Verify tool_compute_rul handles placeholders, JSON, and strings without crashing."""
    # 1. Placeholders
    res1 = agent.tool_compute_rul("avg_temp, total_spikes, operating_hours")
    data1 = json.loads(res1)
    assert "rul_hours" in data1
    assert data1["rul_hours"] is not None

    # 2. Empty string
    res2 = agent.tool_compute_rul("")
    data2 = json.loads(res2)
    assert "rul_hours" in data2

    # 3. JSON format
    res3 = agent.tool_compute_rul(json.dumps({"avg_temp": 50.0, "total_spikes": 2, "operating_hours": 10.0}))
    data3 = json.loads(res3)
    assert data3["health_percentage"] is not None
    assert data3["health_percentage"] > 90.0

    # 4. Comma-separated values
    res4 = agent.tool_compute_rul("48.5, 3, 24.0")
    data4 = json.loads(res4)
    assert data4["rul_hours"] is not None


def test_expert_synthesizer_with_directive():
    """Verify deterministic expert synthesizer includes custom directive and ground truth."""
    metrics = {
        "avg_temp": 62.5,
        "peak_temp": 88.0,
        "total_spikes_over_85": 4,
        "throttling_events": 1,
        "operating_hours": 50.0,
        "reading_count": 200,
        "voltage_v": 1.25,
        "cpu_usage_avg": 45.0,
        "data_quality": "EXCELLENT",
        "last_timestamp": "2026-11-03T12:00:00Z",
    }
    rul = {
        "rul_hours": 38000.0,
        "rul_days": 1583.3,
        "health_percentage": 94.2,
        "degradation_percentage": 5.8,
        "risk_level": "LOW",
        "prediction_method": "PHYSICS_BASED",
        "confidence_interval": {"lower_hours": 20000.0, "upper_hours": 45000.0},
        "limitations": ["Test baseline limitation"],
    }

    report = agent.synthesize_expert_diagnostic_report(
        "node-alpha",
        metrics,
        rul,
        directive="Investigate thermal excursions and heatsink performance.",
    )

    assert "NODE-ALPHA" in report
    assert "INVESTIGATION DIRECTIVE: Investigate thermal excursions and heatsink performance." in report
    assert "SECTION 1: GROUND-TRUTH TELEMETRY OBSERVATIONS" in report
    assert "62.50 °C" in report
    assert "38000.0 hours" in report


def test_run_diagnostic_agent_with_prompt_offline():
    """Verify run_diagnostic_agent executes cleanly when Ollama is offline."""
    with patch("agent.detect_ollama_model", return_value=None):
        report = agent.run_diagnostic_agent("target-cpu", prompt="Evaluate cooling efficiency")
        assert len(report) > 200
        assert "HARDWARE DIAGNOSTIC & PROGNOSTIC REPORT" in report
        assert "INVESTIGATION DIRECTIVE: Evaluate cooling efficiency" in report


def test_run_diagnostic_agent_with_mocked_ollama():
    """Verify run_diagnostic_agent returns LLM generated analysis when Ollama responds."""
    mock_llm_response = (
        "# Diagnostic Report: TARGET-CPU\n\n"
        "## 1. Observations\nObserved nominal telemetry.\n\n"
        "## 2. Root Cause Analysis\nNo thermal fatigue detected.\n\n"
        "## 3. Maintenance Roadmap\nContinue 24/7 monitoring."
    )
    with patch("agent.detect_ollama_model", return_value="dolphin3-cyber"):
        with patch("agent.query_ollama", return_value=mock_llm_response):
            report = agent.run_diagnostic_agent("target-cpu", prompt="Comprehensive analysis")
            assert report == mock_llm_response


def test_dashboard_agent_endpoints():
    """Verify /api/agent/status and /api/agent/diagnose endpoints in dashboard."""
    # 1. Status endpoint
    res_status = client.get("/api/agent/status")
    assert res_status.status_code == 200
    st = res_status.json()
    assert st["agent_ready"] is True
    assert "active_model" in st
    assert len(st["tools_connected"]) >= 3

    # 2. Diagnose endpoint with prompt only (frontend style)
    res_diag = client.post("/api/agent/diagnose", json={"prompt": "Audit thermal cycles and RUL"})
    assert res_diag.status_code == 200
    diag = res_diag.json()
    assert diag["status"] == "success"
    assert "response" in diag
    assert "report" in diag
    assert diag["device_id"] == "target-cpu"
    assert len(diag["response"]) > 100
    assert "timestamp" in diag

    # 3. Diagnose endpoint with explicit device_id and prompt
    res_diag2 = client.post("/api/agent/diagnose", json={"device_id": "target-cpu", "prompt": "Check mitigations"})
    assert res_diag2.status_code == 200
    diag2 = res_diag2.json()
    assert diag2["status"] == "success"
    assert len(diag2["response"]) > 100
