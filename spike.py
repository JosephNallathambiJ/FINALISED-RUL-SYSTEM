#!/usr/bin/env python3
"""Lightweight threshold spike monitor for live system metrics.

This script checks common system values (CPU, memory, temperature, and battery)
while running, and prints a timestamped alert anytime a value goes above its
configured threshold. It keeps monitoring until Ctrl+C is pressed.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import psutil
except ImportError as exc:  # pragma: no cover
    raise SystemExit("psutil is required. Install it with: pip install psutil") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch for threshold spikes in system metrics.")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between checks.")
    parser.add_argument("--steps", type=int, default=0, help="Number of checks to run before exiting. 0 means continuous until Ctrl+C.")
    parser.add_argument("--cpu-threshold", type=float, default=80.0, help="CPU % threshold.")
    parser.add_argument("--memory-threshold", type=float, default=85.0, help="Memory % threshold.")
    parser.add_argument("--temp-threshold", type=float, default=70.0, help="Thermal threshold in C.")
    parser.add_argument("--battery-threshold", type=float, default=20.0, help="Battery low warning threshold; this is shown as a threshold event when battery falls below it.")
    return parser.parse_args()


def current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_temperature_c() -> Optional[float]:
    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, OSError):
        temps = {}

    if temps:
        values: List[float] = []
        for sensor_entries in temps.values():
            for entry in sensor_entries:
                current = getattr(entry, "current", None)
                if current is not None:
                    values.append(float(current))
        if values:
            return max(values)

    thermal_root = "/sys/class/thermal"
    if os.path.isdir(thermal_root):
        candidates: List[float] = []
        for node in os.listdir(thermal_root):
            temp_path = os.path.join(thermal_root, node, "temp")
            if os.path.isfile(temp_path):
                try:
                    with open(temp_path, "r", encoding="utf-8") as handle:
                        value = float(handle.read().strip())
                    candidates.append(value / 1000.0)
                except (OSError, ValueError):
                    continue
        if candidates:
            return max(candidates)

    return None


def get_battery_percent() -> Optional[float]:
    battery = psutil.sensors_battery()
    if battery is None:
        return None
    return float(battery.percent)


def snapshot() -> Dict[str, object]:
    data: Dict[str, object] = {
        "timestamp": current_time(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent,
        "temperature_c": get_temperature_c(),
        "battery_percent": get_battery_percent(),
    }
    return data


def detect_spikes(values: Dict[str, object], cpu_threshold: float, memory_threshold: float, temp_threshold: float, battery_threshold: float) -> List[str]:
    events: List[str] = []

    cpu = values.get("cpu_percent")
    if cpu is not None and float(cpu) > cpu_threshold:
        events.append(f"CPU above threshold: {cpu:.1f}% > {cpu_threshold:.1f}%")

    memory = values.get("memory_percent")
    if memory is not None and float(memory) > memory_threshold:
        events.append(f"Memory above threshold: {memory:.1f}% > {memory_threshold:.1f}%")

    temp = values.get("temperature_c")
    if temp is not None and float(temp) > temp_threshold:
        events.append(f"Temperature above threshold: {temp:.1f}C > {temp_threshold:.1f}C")

    battery = values.get("battery_percent")
    if battery is not None and float(battery) < battery_threshold:
        events.append(f"Battery below threshold: {battery:.1f}% < {battery_threshold:.1f}%")

    return events


def print_snapshot(label: str, values: Dict[str, object]) -> None:
    print(f"[{label}] CPU={values.get('cpu_percent')}% | RAM={values.get('memory_percent')}% | Temp={values.get('temperature_c')}C | Battery={values.get('battery_percent')}%")


def monitor_loop(args: argparse.Namespace) -> int:
    cycle = 0
    stop_requested = False

    def handle_interrupt(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    print("=== THRESHOLD SPIKE MONITOR ===")
    print(f"Interval: {args.interval}s | CPU > {args.cpu_threshold}% | RAM > {args.memory_threshold}% | Temp > {args.temp_threshold}C")
    print("Press Ctrl+C to stop.\n")

    try:
        while not stop_requested:
            cycle += 1
            values = snapshot()
            current = values["timestamp"]
            print_snapshot(current, values)

            events = detect_spikes(
                values,
                cpu_threshold=args.cpu_threshold,
                memory_threshold=args.memory_threshold,
                temp_threshold=args.temp_threshold,
                battery_threshold=args.battery_threshold,
            )

            if events:
                print("[SPIKE] " + " | ".join(events))
            else:
                print("[OK] all monitored values are within threshold.")

            if args.steps and cycle >= args.steps:
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass

    print("\nMonitoring stopped.")
    return 0


def main() -> int:
    args = parse_args()
    return monitor_loop(args)


if __name__ == "__main__":
    sys.exit(main())
