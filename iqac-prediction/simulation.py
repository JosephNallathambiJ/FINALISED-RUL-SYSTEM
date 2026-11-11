#!/usr/bin/env python3
"""Run monitoring-agent simulation cycles and calculate prediction-side RUL."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agent import run_formula_rul_agent
from monitor_client import sync_daily_records

ROOT = Path(__file__).resolve().parents[1]
FORMULA_PATH = Path(__file__).with_name("formula.json")


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_telemetry(result: dict[str, Any]) -> dict[str, float]:
    telemetry = result.get("telemetry", {})
    temperature = result.get("temperature", {})
    fan = result.get("fan_data", {})
    belief = result.get("belief_state", {})
    return {
        "cpu_temp": number(
            temperature.get("temperature_celsius"),
            number(result.get("temperature_c"), number(belief.get("temperature_c"))),
        ),
        "fan_rpm": number(result.get("current_rpm"), number(fan.get("rpm"))),
        "cpu_usage": number(telemetry.get("cpu_percent")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate monitoring agents and calculate RUL")
    parser.add_argument("--monitor-url", required=True, help="Monitoring device URL, e.g. http://192.168.1.20:8001")
    parser.add_argument("--device-id", required=True)
    args = parser.parse_args()
    try:
        sync_result = sync_daily_records(args.monitor_url, args.device_id)
        from database import get_device_metrics
        metrics = get_device_metrics(args.device_id)
        metrics.update({
            "total_spikes": metrics.pop("total_spikes_over_85"),
            "recorded_days": sync_result["daily_records"],
            "total_readings": sync_result["total_readings"],
        })
        formula = json.loads(FORMULA_PATH.read_text(encoding="utf-8"))
        report = run_formula_rul_agent(metrics, formula)
        print(json.dumps(report, indent=2))
        return 0
    except (RuntimeError, json.JSONDecodeError, KeyError, OSError, ZeroDivisionError) as error:
        print(f"Prediction paused: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
