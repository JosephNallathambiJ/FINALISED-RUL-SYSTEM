"""Tests for Physics-Based RUL Engine and Degradation Modeling."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

PRED_DIR = Path(__file__).resolve().parents[1] / "iqac-prediction"
sys.path.insert(0, str(PRED_DIR))

from rul_engine import (
    COMPONENT_PROFILES,
    ComponentProfile,
    DataQuality,
    PhysicsRULEngine,
    RiskLevel,
    calculate_arrhenius_af,
    calculate_thermal_cycle_damage,
    calculate_voltage_af,
    compute_rul,
    get_profile,
)


def test_normal_cpu_temperature():
    """Test Case 1: Normal steady-state CPU temperature (45°C) yields low degradation and long RUL."""
    engine = PhysicsRULEngine(profile=get_profile("desktop"))
    res = engine.evaluate(avg_temp=45.0, total_spikes_over_85=0, operating_hours=100.0, reading_count=50)

    assert res.rul_hours is not None
    assert res.rul_hours > 30000.0
    assert res.health_percentage is not None and res.health_percentage > 95.0
    assert res.degradation_percentage is not None and res.degradation_percentage < 5.0
    assert res.risk_level == RiskLevel.LOW.value
    assert res.data_quality == DataQuality.EXCELLENT.value


def test_sustained_high_temperature():
    """Test Case 2: Sustained high temperature (90°C) accelerates Arrhenius damage significantly."""
    engine = PhysicsRULEngine(profile=get_profile("desktop"))
    res_normal = engine.evaluate(avg_temp=45.0, total_spikes_over_85=0, operating_hours=500.0, reading_count=50)
    res_hot = engine.evaluate(avg_temp=90.0, total_spikes_over_85=0, operating_hours=500.0, reading_count=50)

    assert res_hot.rul_hours is not None
    assert res_normal.rul_hours is not None
    assert res_hot.rul_hours < res_normal.rul_hours
    assert res_hot.degradation_percentage > res_normal.degradation_percentage
    # At 90°C sustained, Arrhenius acceleration factor is much greater than 1
    af_90 = calculate_arrhenius_af(90.0, ref_temp_c=45.0, activation_energy_ev=0.7)
    assert af_90 > 10.0


def test_repeated_thermal_spikes():
    """Test Case 3: Repeated thermal excursions (>85°C) accumulate Coffin-Manson cycle fatigue."""
    engine = PhysicsRULEngine(profile=get_profile("desktop"))
    res_no_spikes = engine.evaluate(avg_temp=55.0, total_spikes_over_85=0, operating_hours=200.0, reading_count=50)
    res_with_spikes = engine.evaluate(avg_temp=55.0, total_spikes_over_85=25, operating_hours=200.0, reading_count=50)

    assert res_with_spikes.degradation_percentage > res_no_spikes.degradation_percentage
    assert res_with_spikes.rul_hours < res_no_spikes.rul_hours
    assert res_with_spikes.risk_level in (RiskLevel.MODERATE.value, RiskLevel.HIGH.value)


def test_missing_temperature_sensor():
    """Test Case 4: Missing/null temperature sensor refuses to fabricate fake RUL."""
    engine = PhysicsRULEngine()
    res = engine.evaluate(avg_temp=None, total_spikes_over_85=0, operating_hours=100.0)

    assert res.rul_hours is None
    assert res.rul_days is None
    assert res.health_percentage is None
    assert res.risk_level == RiskLevel.UNKNOWN.value
    assert res.data_quality == DataQuality.INSUFFICIENT.value
    assert any("null or unavailable" in lim for lim in res.limitations)


def test_zero_or_negative_sensor_readings():
    """Test Case 5: Zero, negative, or unphysically extreme readings are rejected."""
    engine = PhysicsRULEngine()

    for invalid_temp in [0.0, -15.0, 150.0]:
        res = engine.evaluate(avg_temp=invalid_temp, total_spikes_over_85=0, operating_hours=100.0)
        assert res.rul_hours is None
        assert res.risk_level == RiskLevel.UNKNOWN.value
        assert res.data_quality == DataQuality.INSUFFICIENT.value
        assert any("outside physical operating limits" in lim for lim in res.limitations)


def test_missing_voltage_information():
    """Test Case 6: Missing voltage falls back gracefully to nominal and logs limitation."""
    engine = PhysicsRULEngine()
    res = engine.evaluate(avg_temp=50.0, total_spikes_over_85=0, operating_hours=100.0, voltage_v=None, reading_count=30)

    assert res.rul_hours is not None
    assert any("voltage sensor unavailable" in lim for lim in res.limitations)


def test_missing_historical_data_and_short_duration():
    """Test Cases 7 & 8: Short monitoring duration (< 1h) or low readings is flagged as DEGRADED."""
    engine = PhysicsRULEngine()
    res = engine.evaluate(avg_temp=50.0, total_spikes_over_85=0, operating_hours=0.2, reading_count=5)

    assert res.data_quality == DataQuality.DEGRADED.value
    assert any("duration is very short" in lim for lim in res.limitations)


def test_rul_boundary_conditions():
    """Test Case 17: Boundary conditions (extreme damage, zero RUL clamping, initial life)."""
    engine = PhysicsRULEngine()

    # Extreme accumulated stress leading to total damage >= 1.0
    res_failed = engine.evaluate(avg_temp=115.0, total_spikes_over_85=200, operating_hours=60000.0)
    assert res_failed.rul_hours == 0.0
    assert res_failed.health_percentage == 0.0
    assert res_failed.degradation_percentage == 100.0
    assert res_failed.risk_level == RiskLevel.CRITICAL.value

    # Upper bound clamping (cannot exceed 1.5x base life)
    res_cool = engine.evaluate(avg_temp=25.0, total_spikes_over_85=0, operating_hours=1.0)
    assert res_cool.rul_hours <= engine.profile.base_life_hours * 1.5


def test_component_profiles():
    """Test component-specific reliability parameters for server vs desktop vs laptop."""
    p_desktop = get_profile("desktop")
    p_server = get_profile("server")
    p_laptop = get_profile("laptop")
    p_iot = get_profile("iot")

    assert p_server.base_life_hours > p_desktop.base_life_hours
    assert p_server.ref_temp_c < p_laptop.ref_temp_c
    assert p_laptop.max_operating_temp_c >= 100.0
    assert not p_desktop.is_calibrated


def test_backward_compatibility_compute_rul():
    """Ensure legacy compute_rul() returns expected dictionary format."""
    res = compute_rul(avg_temp=60.0, total_spikes_over_85=5, operating_hours=150.0)
    assert "estimated_rul_hours" in res
    assert "health_score_pct" in res
    assert "risk_level" in res
    assert "component" in res
    assert "data_quality" in res

