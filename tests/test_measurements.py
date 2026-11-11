"""Tests for automated telemetry measurement supervisor and hardware collectors."""

import time
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from control.measurement_manager import MeasurementManager, collect_hardware_telemetry


def test_collect_hardware_telemetry_schema():
    """Verify hardware telemetry collection adheres to schema without fabricating values."""
    snap = collect_hardware_telemetry()
    
    assert "cpu_util" in snap
    assert "cpu_temp" in snap
    assert "cpu_freq_mhz" in snap
    assert "memory_util_pct" in snap
    assert "timestamp" in snap
    assert "hardware_id" in snap

    # CPU utilization must be float between 0 and 100
    if snap["cpu_util"] is not None:
        assert 0.0 <= snap["cpu_util"] <= 100.0

    # Memory utilization must be float between 0 and 100
    if snap["memory_util_pct"] is not None:
        assert 0.0 <= snap["memory_util_pct"] <= 100.0

    # Temperature if present must be physically reasonable, else None (NEVER 0.0)
    if snap["cpu_temp"] is not None:
        assert 10.0 <= snap["cpu_temp"] <= 130.0
    else:
        # Must be explicitly None, not fabricated 0
        assert snap["cpu_temp"] is None


def test_measurement_manager_lifecycle():
    """Verify Start, Stop, Pause, Resume, and Interval changes in MeasurementManager."""
    ticks = []

    def on_tick(snap):
        ticks.append(snap)

    mgr = MeasurementManager(sampling_interval=0.2, on_measurement=on_tick)
    assert not mgr.is_running
    assert not mgr.is_paused

    # Start
    mgr.start()
    assert mgr.is_running
    time.sleep(0.5)
    assert len(ticks) >= 1

    # Pause
    mgr.pause()
    assert mgr.is_paused
    count_before = len(ticks)
    time.sleep(0.4)
    # No new ticks while paused
    assert len(ticks) == count_before

    # Resume
    mgr.resume()
    assert not mgr.is_paused
    time.sleep(0.4)
    assert len(ticks) > count_before

    # Change interval
    mgr.set_interval(0.5)
    assert mgr.sampling_interval == 0.5

    # Stop
    mgr.stop()
    assert not mgr.is_running


def test_measurement_circular_buffer():
    """Verify in-memory circular buffer retains recent snapshots up to max_buffer."""
    mgr = MeasurementManager(sampling_interval=1.0)
    for i in range(15):
        mgr._record_snapshot({"val": i, "timestamp": time.time()})

    snaps = mgr.get_recent_snapshots(5)
    assert len(snaps) == 5
    assert snaps[-1]["val"] == 14

