"""Tests for Monitoring, Telemetry Validation, Database Persistence, and Two-Machine Sync."""

from __future__ import annotations

import datetime
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT_DIR / "iqac-prediction"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PRED_DIR))

import recorder
from database import (
    get_device_metrics,
    get_device_spike_summary,
    init_db,
    insert_daily_record,
    insert_telemetry,
    update_sync_state,
)
from monitor_client import normalize_status, sync_daily_records


def test_recorder_telemetry_sanitization():
    """Test Case 9: Recorder sanitizes extreme, negative, or invalid telemetry."""
    # Test CPU clamping: negative CPU should clamp to 0, >100 should clamp to 100
    # Test Temperature rejection: 0°C or negative should be rejected
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    recorder.record_reading(
        temperature_c=0.0,
        cpu_usage=-25.0,
        fan_rpm=-500.0,
        recorded_at=now,
    )
    records = recorder.get_records()
    today = now[:10]
    today_rec = next((r for r in records["daily"] if r["day"] == today), None)
    assert today_rec is not None
    assert today_rec["average_cpu_usage"] >= 0.0
    assert today_rec["average_fan_rpm"] >= 0.0
    # 0.0 temp must not be recorded as valid temperature
    assert today_rec["average_temperature_c"] is None or today_rec["average_temperature_c"] > 0.0


def test_out_of_order_timestamps():
    """Test Case 10: Telemetry recorded out-of-order is aggregated properly without crashing."""
    import uuid
    dev_id = f"test_ooo_{uuid.uuid4().hex[:8]}"
    init_db()

    # Insert newer timestamp then older timestamp
    t_newer = "2026-05-10T12:00:00+00:00"
    t_older = "2026-05-10T10:00:00+00:00"

    insert_telemetry(device_id=dev_id, cpu_temp=48.0, fan_rpm=2200, cpu_usage=35.0, timestamp=t_newer)
    insert_telemetry(device_id=dev_id, cpu_temp=52.0, fan_rpm=2400, cpu_usage=45.0, timestamp=t_older)

    metrics = get_device_metrics(dev_id)
    assert metrics["reading_count"] == 2
    assert metrics["avg_temp"] == 50.0
    assert metrics["operating_hours"] == 2.0  # 12:00 - 10:00 = 2.0 hrs


def test_duplicate_synchronization_prevention():
    """Test Case 11: Repeated synchronization updates existing days without duplicating rows."""
    import uuid
    dev_id = f"test_dup_{uuid.uuid4().hex[:8]}"
    init_db()

    day_record = {
        "day": "2026-06-01",
        "reading_count": 50,
        "average_temperature_c": 44.5,
        "average_fan_rpm": 2100.0,
        "average_cpu_usage": 28.0,
        "spike_count": 0,
        "updated_at": "2026-06-01T23:59:59+00:00",
    }

    # First sync
    insert_daily_record(dev_id, day_record)
    metrics_1 = get_device_metrics(dev_id)
    assert metrics_1["reading_count"] == 50

    # Duplicate sync with identical day (e.g. repeated pull)
    insert_daily_record(dev_id, day_record)
    metrics_2 = get_device_metrics(dev_id)
    assert metrics_2["reading_count"] == 50  # Must NOT be 100


def test_normalize_status_validation():
    """Verify monitor_client normalizes incoming telemetry and discards corrupt sensor values."""
    raw_status = {
        "cpu_temp": 0.0,  # Invalid
        "current_rpm": "2400",
        "cpu_usage": 150.0,  # Out of range, should clamp to 100
        "voltage_v": 1.15,
        "is_spike": False,
        "timestamp": 1700000000.0,
    }
    normalized = normalize_status(raw_status, "test_client_dev")
    assert normalized["cpu_temp"] is None  # 0.0 is filtered out
    assert normalized["cpu_usage"] == 100.0
    assert normalized["fan_rpm"] == 2400
    assert normalized["voltage_v"] == 1.15


def test_network_retry_and_timeout():
    """Test Case 12: Network disconnection triggers retry logic and raises ConnectionError on exhaustion."""
    with pytest.raises(ConnectionError) as exc_info:
        sync_daily_records(
            monitor_url="http://127.0.0.1:59999",  # unreachable port
            device_id="test_unreachable",
            timeout=0.1,
            max_retries=2,
        )
    assert "Failed to connect" in str(exc_info.value)


def test_hardware_identity_capture():
    """Verify hardware identification properties are correctly discovered."""
    hw = recorder.get_hardware_identity()
    assert "platform" in hw
    assert "architecture" in hw
    assert "cpu_count_physical" in hw
    assert hw["cpu_count_physical"] >= 1
