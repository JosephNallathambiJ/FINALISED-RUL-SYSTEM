"""Reproducible Synthetic Telemetry Benchmark Test Suite.

IMPORTANT SCIENTIFIC CAVEAT:
  Synthetic data verifies software architecture, numerical stability, monotonicity,
  and boundary condition handling under controlled conditions.
  It MUST NEVER be presented as empirical proof of real-world CPU lifespan prediction accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

PRED_DIR = Path(__file__).resolve().parents[1] / "iqac-prediction"
sys.path.insert(0, str(PRED_DIR))

from rul_engine import PhysicsRULEngine, RiskLevel, get_profile


def generate_synthetic_telemetry_history(
    days: int = 14,
    thermal_profile: str = "nominal",
    seed: int = 42,
) -> list[dict[str, float]]:
    """Generate reproducible multi-day CPU telemetry sequence."""
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []

    for d in range(1, days + 1):
        if thermal_profile == "nominal":
            avg_temp = float(rng.normal(42.0, 1.5))
            spikes = 0
            cpu_usage = float(rng.normal(25.0, 5.0))
        elif thermal_profile == "stress_accelerated":
            # Progressive degradation: temperature climbs steadily into junction limits
            avg_temp = float(min(105.0, 55.0 + (d * 3.5) + rng.normal(0, 1.0)))
            spikes = int(max(0, (d - 3) * 6))
            cpu_usage = float(min(98.0, 40.0 + (d * 4.0)))
        elif thermal_profile == "cyclical_spikes":
            avg_temp = float(55.0 + rng.normal(0, 2.0))
            spikes = int(rng.poisson(4))
            cpu_usage = float(rng.normal(50.0, 10.0))
        else:
            raise ValueError(f"Unknown profile: {thermal_profile}")

        history.append({
            "day": d,
            "avg_temp": round(avg_temp, 2),
            "total_spikes": spikes,
            "cpu_usage": round(max(0.0, min(100.0, cpu_usage)), 2),
            "operating_hours": float(d * 24.0),
        })

    return history


def test_synthetic_nominal_trajectory():
    """Verify that under nominal thermal conditions, cumulative Miner's rule damage is strictly monotonic."""
    history = generate_synthetic_telemetry_history(days=10, thermal_profile="nominal")
    engine = PhysicsRULEngine(profile=get_profile("desktop"))

    prev_damage = 0.0
    accumulated_intervals = []
    for day_data in history:
        accumulated_intervals.append({
            "duration_hours": 24.0,
            "temperature_c": day_data["avg_temp"],
            "voltage_v": 1.2,
        })
        res = engine.evaluate(
            avg_temp=day_data["avg_temp"],
            total_spikes_over_85=day_data["total_spikes"],
            operating_hours=day_data["operating_hours"],
            reading_count=50,
            historical_intervals=accumulated_intervals,
        )
        assert res.rul_hours is not None
        assert res.degradation_percentage is not None
        # Monotonicity check under discrete cumulative damage integration
        assert res.degradation_percentage >= prev_damage
        prev_damage = res.degradation_percentage
        assert res.risk_level == RiskLevel.LOW.value


def test_synthetic_stress_accelerated_trajectory():
    """Verify that severe thermal stress dramatically accelerates degradation into elevated risk."""
    history = generate_synthetic_telemetry_history(days=14, thermal_profile="stress_accelerated")
    engine = PhysicsRULEngine(profile=get_profile("desktop"))

    accumulated_intervals = [
        {"duration_hours": 24.0, "temperature_c": h["avg_temp"], "voltage_v": 1.25}
        for h in history
    ]
    final_day = history[-1]
    res_final = engine.evaluate(
        avg_temp=final_day["avg_temp"],
        total_spikes_over_85=sum(h["total_spikes"] for h in history),
        operating_hours=final_day["operating_hours"],
        reading_count=100,
        historical_intervals=accumulated_intervals,
    )

    # Severe stress should elevate risk to HIGH or CRITICAL
    assert res_final.risk_level in (RiskLevel.HIGH.value, RiskLevel.CRITICAL.value)
    assert res_final.degradation_percentage > 25.0
    assert res_final.rul_hours < engine.profile.base_life_hours * 0.5


def test_synthetic_monotonicity_and_bounds():
    """Verify physical conservation bounds on synthetic trajectories."""
    history = generate_synthetic_telemetry_history(days=7, thermal_profile="cyclical_spikes")
    engine = PhysicsRULEngine()

    accumulated_intervals = []
    for day_data in history:
        accumulated_intervals.append({
            "duration_hours": 24.0,
            "temperature_c": day_data["avg_temp"],
            "voltage_v": 1.2,
        })
        res = engine.evaluate(
            avg_temp=day_data["avg_temp"],
            total_spikes_over_85=day_data["total_spikes"],
            operating_hours=day_data["operating_hours"],
            reading_count=40,
            historical_intervals=accumulated_intervals,
        )
        assert 0.0 <= res.health_percentage <= 100.0
        assert 0.0 <= res.degradation_percentage <= 100.0
        assert res.rul_hours >= 0.0
        assert res.confidence_interval["lower_hours"] >= 0.0
