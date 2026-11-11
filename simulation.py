#!/usr/bin/env python3
"""Cross-platform smoke-test simulation for the AI agents.

This script uses the machine's built-in operating-system sensors via psutil
(CPU, RAM, battery, temperature where supported) and runs a short live test
through each agent's normal decision loop.

It is designed to work on Linux, Windows, and macOS without requiring
hardware-specific sensor drivers.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import psutil

ROOT = Path(__file__).resolve().parent
SHARED_TELEMETRY_PATH = ROOT / "iqac-prediction" / "shared_telemetry.json"


def publish_prediction_telemetry(status: str, cycle: int, readings: list[dict[str, Any]]) -> None:
    """Publish the latest monitoring heartbeat for the separate prediction system."""
    from recorder import get_records
    payload = {
        "monitoring_running": status == "running",
        "status": status,
        "cycle": cycle,
        "timestamp": time.time(),
        "readings": readings,
        "records": get_records(),
    }
    temporary_path = SHARED_TELEMETRY_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(SHARED_TELEMETRY_PATH)


def system_snapshot() -> Dict[str, Any]:
    """Collect system health data from built-in OS sensors via psutil."""
    snapshot: Dict[str, Any] = {
        "platform": platform.system(),
        "release": platform.release(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory": {
            "percent": psutil.virtual_memory().percent,
            "used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
            "total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        },
        "disk": {
            "percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
        },
        "battery": None,
        "temperatures": None,
    }

    try:
        battery = psutil.sensors_battery()
        if battery is not None:
            snapshot["battery"] = {
                "percent": battery.percent,
                "plugged_in": bool(battery.power_plugged),
                "seconds_left": battery.secsleft,
            }
    except Exception:
        snapshot["battery"] = None

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            reduced = {}
            for name, entries in temps.items():
                cleaned = []
                for entry in entries:
                    cleaned.append({
                        "label": getattr(entry, "label", ""),
                        "current": getattr(entry, "current", None),
                        "high": getattr(entry, "high", None),
                        "critical": getattr(entry, "critical", None),
                    })
                reduced[name] = cleaned
            snapshot["temperatures"] = reduced
    except Exception:
        snapshot["temperatures"] = None

    return snapshot


def ensure_agent_path(agent_name: str) -> str:
    agent_dir = ROOT / agent_name
    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent directory not found: {agent_dir}")
    return str(agent_dir)


def get_agent_factory(agent_name: str):
    """Import the correct agent class and config loader for each agent."""
    if agent_name == "agent1":
        from agent1_core.agent import ModelUtilityLearningAgent
        from agent1_core.config import load_config
        return ModelUtilityLearningAgent, load_config
    if agent_name == "agent2":
        from agent2_core.agent import FanCoolingAgent
        from agent2_core.config import load_config
        return FanCoolingAgent, load_config
    if agent_name == "agent3":
        from agent3_core.agent import SystemVoltageAgent
        from agent3_core.config import load_config
        return SystemVoltageAgent, load_config
    if agent_name == "agent4":
        from agent4_core.agent import CpuPerformanceAgent
        from agent4_core.config import load_config
        return CpuPerformanceAgent, load_config
    raise ValueError(f"Unsupported agent: {agent_name}")


def run_agent_smoke(agent_name: str, steps: int = 2) -> Dict[str, Any]:
    agent_dir = ensure_agent_path(agent_name)
    original_cwd = os.getcwd()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, agent_dir)

    try:
        os.chdir(agent_dir)
        factory, load_config = get_agent_factory(agent_name)
        config = load_config("config.yaml")
        agent = factory(config=config)

        history: List[Dict[str, Any]] = []
        for _ in range(max(1, steps)):
            result = agent.step()
            history.append({
                "step": result.get("step"),
                "profile": result.get("profile"),
                "summary": summarize_agent_result(agent_name, result),
                "raw_result": result,
            })

        return {
            "agent": agent_name,
            "status": "OK",
            "platform": platform.system(),
            "steps_run": len(history),
            "history": history,
        }
    except Exception as exc:
        return {
            "agent": agent_name,
            "status": "ERROR",
            "platform": platform.system(),
            "error": str(exc),
        }
    finally:
        os.chdir(original_cwd)


def summarize_agent_result(agent_name: str, result: Dict[str, Any]) -> str:
    if agent_name == "agent1":
        temp = result.get("temperature", {}).get("temperature_celsius")
        cpu = result.get("telemetry", {}).get("cpu_percent")
        status = result.get("belief_state", {}).get("thermal_status")
        action = result.get("action", {}).get("executed_action")
        return f"temp={temp}C cpu={cpu}% status={status} action={action}"
    if agent_name == "agent2":
        temp = result.get("temperature_c")
        rpm = result.get("current_rpm")
        pwm = result.get("target_pwm_pct")
        return f"temp={temp}C rpm={rpm} pwm={pwm}%"
    if agent_name == "agent3":
        rails = result.get("belief", {})
        return f"vcore={rails.get('vcore_volts')}V status={rails.get('voltage_status')}"
    if agent_name == "agent4":
        freq = result.get("freq_data", {}).get("freq_mhz")
        cpu = result.get("telemetry", {}).get("cpu_percent")
        gov = result.get("governor", {}).get("governor_mode")
        return f"freq={freq}MHz cpu={cpu}% governor={gov}"
    return str(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cross-platform 24/7 monitoring smoke test for the project agents.")
    parser.add_argument(
        "--agents",
        nargs="*",
        default=["agent1", "agent2", "agent3", "agent4"],
        help="Agent names to monitor. Defaults to all runtime agents.",
    )
    parser.add_argument("--steps", type=int, default=0, help="Number of monitor cycles per agent. Use 0 (default) for continuous monitoring until Ctrl+C.")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between monitor cycles.")
    return parser.parse_args()


def has_threshold_alert(agent_name: str, result: Dict[str, Any]) -> bool:
    if agent_name == "agent1":
        real_temp = result.get("temperature", {}).get("temperature_celsius")
        threshold = result.get("belief_state", {}).get("thermal_status")
        return bool(real_temp is not None and threshold in {"WARM", "WARNING", "CRITICAL", "OVERHEAT"})
    if agent_name == "agent2":
        temp = result.get("temperature_c")
        return bool(temp is not None and temp >= 60)
    if agent_name == "agent3":
        rails = result.get("belief", {})
        status = rails.get("voltage_status")
        return bool(status in {"UNDER_VOLTAGE_SAG", "OVER_VOLTAGE_SURGE", "BROWNOUT_RISK", "CRITICAL_BATTERY"})
    if agent_name == "agent4":
        return bool(result.get("governor", {}).get("governor_mode") in {"PERFORMANCE", "BALANCED"})
    return False


def monitor_loop(
    args: argparse.Namespace,
    record_readings: bool = False,
    publish_shared_file: bool = True,
) -> int:
    print("=== 24/7 SYSTEM MONITORING ===")
    print("Press Ctrl+C to stop monitoring.")
    print(f"Agents: {', '.join(args.agents)}")
    print(f"Monitor interval: {args.interval}s")

    cycle = 0
    if publish_shared_file:
        publish_prediction_telemetry("running", cycle, [])
    try:
        while True:
            cycle += 1
            print(f"\n--- Monitor cycle #{cycle} ---")
            snapshot = system_snapshot()
            print(f"[{time.strftime('%H:%M:%S')}] Platform: {snapshot['platform']} {snapshot['release']}")
            print(f"CPU: {snapshot['cpu_percent']}%, Memory: {snapshot['memory']['percent']}%")
            if snapshot['battery']:
                print(f"Battery: {snapshot['battery']}")

            shared_readings: list[dict[str, Any]] = []
            cycle_temp: Optional[float] = None
            cycle_cpu: float = float(snapshot["cpu_percent"] or 0.0)
            cycle_fan: float = 0.0
            cycle_voltage: Optional[float] = None
            cycle_freq: Optional[float] = None
            cycle_is_spike: bool = False
            cycle_throttled: bool = False
            cycle_details: dict[str, Any] = {}

            for agent_name in args.agents:
                result = run_agent_smoke(agent_name, steps=1)
                if result["status"] == "OK":
                    raw = result["history"][0].get("raw_result", {}) if result["history"] else {}
                    summary = result["history"][0]["summary"] if result["history"] else "no data"
                    print(f"[{agent_name}] {summary}")
                    alert_triggered = has_threshold_alert(agent_name, result["history"][0] if result["history"] else {})
                    if alert_triggered:
                        print(f"[ALERT] {agent_name} threshold breach detected.")
                        cycle_is_spike = True

                    shared_readings.append({
                        "agent": agent_name,
                        "result": raw,
                    })

                    # Extract sensor telemetry without polluting with zeros
                    if agent_name == "agent1":
                        t_val = raw.get("temperature", {}).get("temperature_celsius", raw.get("temperature_c"))
                        if t_val is not None and float(t_val) > 0:
                            cycle_temp = float(t_val)
                        cycle_details["agent1_status"] = raw.get("belief_state", {}).get("thermal_status")
                        if cycle_details["agent1_status"] in {"CRITICAL", "OVERHEAT"}:
                            cycle_throttled = True

                    elif agent_name == "agent2":
                        t_val = raw.get("temperature_c")
                        if cycle_temp is None and t_val is not None and float(t_val) > 0:
                            cycle_temp = float(t_val)
                        rpm_val = raw.get("current_rpm", raw.get("fan_data", {}).get("rpm"))
                        if rpm_val is not None:
                            cycle_fan = float(rpm_val)

                    elif agent_name == "agent3":
                        v_val = raw.get("belief", {}).get("vcore_volts")
                        if v_val is not None:
                            cycle_voltage = float(v_val)

                    elif agent_name == "agent4":
                        f_val = raw.get("freq_data", {}).get("freq_mhz")
                        if f_val is not None:
                            cycle_freq = float(f_val)
                        gov = raw.get("governor", {}).get("governor_mode")
                        if gov == "POWERSAVE":
                            cycle_throttled = True
                else:
                    print(f"[{agent_name}] ERROR: {result['error']}")

            if record_readings:
                from recorder import record_reading
                if cycle_temp is not None and cycle_temp >= 85.0:
                    cycle_is_spike = True

                record_reading(
                    temperature_c=cycle_temp,
                    cpu_usage=cycle_cpu,
                    fan_rpm=cycle_fan,
                    voltage_v=cycle_voltage,
                    cpu_freq_mhz=cycle_freq,
                    throttling=cycle_throttled,
                    agent="system_monitor",
                    is_spike=cycle_is_spike,
                    details=cycle_details,
                )

            if publish_shared_file:
                publish_prediction_telemetry("running", cycle, shared_readings)

            if args.steps and cycle >= args.steps:
                print("\nRequested step limit reached. Monitoring stopped.")
                if publish_shared_file:
                    publish_prediction_telemetry("stopped", cycle, shared_readings)
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if publish_shared_file:
            publish_prediction_telemetry("stopped", cycle, [])
        print("\nMonitoring stopped by user (Ctrl+C).")
        return 0


def main() -> int:
    args = parse_args()
    return monitor_loop(args, record_readings=True)


if __name__ == "__main__":
    raise SystemExit(main())
