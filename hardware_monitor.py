#!/usr/bin/env python3
"""Run IQAC Agent 1 with a Linux 1-Wire temperature sensor.

This is a small first hardware integration for Raspberry Pi and similar Linux
IoT gateways. The DS18B20 sensor must be visible below /sys/bus/w1/devices.
"""

from __future__ import annotations

import argparse
import glob
import sys
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agent1"))

from agent1_core.agent import ModelUtilityLearningAgent
from agent1_core.config import load_config
from recorder import app as recorder_app
from recorder import record_reading


def find_sensor_file(sensor_id: str | None) -> Path:
    pattern = f"/sys/bus/w1/devices/{sensor_id}/w1_slave" if sensor_id else "/sys/bus/w1/devices/28-*/w1_slave"
    matches = sorted(Path(path) for path in glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            "No DS18B20 sensor found. Enable the Linux w1-gpio and w1-therm modules, "
            "check wiring, or pass --sensor-id for a specific sensor."
        )
    return matches[0]


def read_ds18b20(sensor_file: Path) -> float:
    lines = sensor_file.read_text(encoding="ascii").splitlines()
    if len(lines) < 2 or not lines[0].strip().endswith("YES"):
        raise RuntimeError(f"DS18B20 CRC check failed: {sensor_file}")
    marker = "t="
    if marker not in lines[1]:
        raise RuntimeError(f"DS18B20 temperature value missing: {sensor_file}")
    return float(lines[1].split(marker, 1)[1]) / 1000.0


def run(args: argparse.Namespace) -> int:
    sensor_file = find_sensor_file(args.sensor_id)
    config = load_config(str(ROOT / "agent1" / "config.yaml"))
    agent = ModelUtilityLearningAgent(config=config, sensor_mode="custom")
    agent.register_hardware_temperature_sensor(lambda: read_ds18b20(sensor_file))

    print(f"Using physical temperature sensor: {sensor_file}")
    print(f"Records API: http://{args.host}:{args.port}/records")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            result: dict[str, Any] = agent.step()
            temperature = float(result["temperature"]["temperature_celsius"])
            telemetry = result.get("telemetry", {})
            status = result.get("belief_state", {}).get("thermal_status")
            is_spike = status in {"WARM", "WARNING", "CRITICAL", "OVERHEAT"} or temperature >= args.spike_temperature
            record_reading(
                temperature_c=temperature,
                cpu_usage=float(telemetry.get("cpu_percent", 0.0) or 0.0),
                agent="agent1",
                is_spike=is_spike,
                details={"sensor": str(sensor_file), "thermal_status": status},
            )
            print(f"temp={temperature:.2f}C cpu={telemetry.get('cpu_percent', 0.0):.1f}% status={status}")
        except Exception as error:
            print(f"Sensor/monitoring error: {error}. Retrying on the next cycle.")
        time.sleep(args.interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run IQAC with a physical DS18B20 temperature sensor")
    parser.add_argument("--sensor-id", help="Optional 1-Wire ID, for example 28-00000abcdef")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between readings")
    parser.add_argument("--spike-temperature", type=float, default=85.0, help="Temperature that creates a recorded spike")
    parser.add_argument("--host", default="0.0.0.0", help="Records API bind address")
    parser.add_argument("--port", type=int, default=8001, help="Records API port")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    api_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": recorder_app, "host": args.host, "port": args.port, "log_level": "warning"},
        daemon=True,
    )
    api_thread.start()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\nHardware monitoring stopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
