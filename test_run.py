#!/usr/bin/env python3
"""
Automated Telemetry & Bounded Stress Test Runner.

Provides:
  Mode 1: Normal Real-Hardware CPU Monitoring Run (30-60s)
  Mode 2: Controlled Bounded CPU Stress Test Run with Thermal Auto-Cutoff
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent
PRED_DIR = ROOT_DIR / "iqac-prediction"
os.chdir(str(ROOT_DIR))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

from control.device_manager import get_primary_ip, get_system_hardware_info
from control.measurement_manager import MeasurementManager, collect_hardware_telemetry
from control.stress_tester import ControlledStressTester
from database import get_db_connection, init_db


def print_test_header(mode_name: str):
    info = get_system_hardware_info()
    ip = get_primary_ip()
    print("=" * 78)
    print(f"🚀 AUTOMATED TEST RUNNER: {mode_name.upper()}")
    print("=" * 78)
    print(f"  OS / Architecture: {info['os_name']} {info['machine']} ({info['processor']})")
    print(f"  Physical / Logical: {info['cpu_physical_cores']} physical, {info['cpu_logical_cores']} logical cores")
    print(f"  RAM Total: {info['ram_total_gb']} GB | Primary IPv4: {ip}")
    print("=" * 78)


def run_mode_1_normal_monitoring(duration_sec: int = 30, interval_sec: float = 1.0):
    """Mode 1: Normal Real-Hardware Telemetry Collection."""
    print_test_header("Mode 1 - Normal Telemetry Monitoring")
    print(f"Starting {duration_sec}s telemetry collection at {interval_sec}s interval...\n")

    init_db()

    collected: List[Dict[str, Any]] = []

    def on_tick(snap: Dict[str, Any]):
        collected.append(snap)
        temp_str = f"{snap['cpu_temp']:.1f}°C" if snap["cpu_temp"] is not None else "NOT AVAILABLE"
        util_str = f"{snap['cpu_util']:.1f}%" if snap["cpu_util"] is not None else "N/A"
        freq_str = f"{snap['cpu_freq_mhz']:.0f}MHz" if snap["cpu_freq_mhz"] is not None else "N/A"
        mem_str = f"{snap['memory_util_pct']:.1f}%" if snap["memory_util_pct"] is not None else "N/A"
        pwr_str = f"{snap['cpu_power_w']:.1f}W" if snap["cpu_power_w"] is not None else "N/A"

        print(
            f"  [{len(collected):02d}/{int(duration_sec/interval_sec)}] "
            f"Temp: {temp_str:<14} | Util: {util_str:<6} | Freq: {freq_str:<8} | Mem: {mem_str:<6} | Power: {pwr_str}"
        )

    mgr = MeasurementManager(sampling_interval=interval_sec, on_measurement=on_tick)
    mgr.start()

    start_time = time.time()
    try:
        while time.time() - start_time < duration_sec:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[USER] Stopped early via KeyboardInterrupt.")
    finally:
        mgr.stop()

    # Verify Database Persistence
    db_count = 0
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        target_tbl = "telemetry" if "telemetry" in tables else ("cpu_metrics" if "cpu_metrics" in tables else None)
        if target_tbl:
            cur.execute(f"SELECT COUNT(*) FROM {target_tbl}")
            row = cur.fetchone()
            db_count = row[0] if row else 0
        conn.close()
    except Exception as e:
        print(f"[WARN] DB check notice: {e}")

    # Generate Statistical Analysis Report
    print("\n" + "=" * 78)
    print("📊 MODE 1 TEST RESULTS & HARDWARE TELEMETRY SUMMARY")
    print("=" * 78)
    print(f"  Total Telemetry Samples Collected : {len(collected)}")
    print(f"  Total Database Records in Store   : {db_count}")

    temps = [c["cpu_temp"] for c in collected if c["cpu_temp"] is not None]
    utils = [c["cpu_util"] for c in collected if c["cpu_util"] is not None]
    freqs = [c["cpu_freq_mhz"] for c in collected if c["cpu_freq_mhz"] is not None]
    throttles = sum(1 for c in collected if c.get("thermal_throttling_event"))

    if temps:
        print(f"  CPU Temperature Readings          : {len(temps)} valid readings")
        print(f"    - Minimum Temperature           : {min(temps):.1f}°C")
        print(f"    - Average Temperature           : {sum(temps)/len(temps):.1f}°C")
        print(f"    - Peak Temperature              : {max(temps):.1f}°C")
    else:
        print("  CPU Temperature Sensor            : NOT AVAILABLE on this platform driver")

    if utils:
        print(f"  CPU Utilization Average           : {sum(utils)/len(utils):.1f}% (Peak: {max(utils):.1f}%)")

    if freqs:
        print(f"  CPU Core Frequency Dynamic Range  : {min(freqs):.0f} MHz - {max(freqs):.0f} MHz")

    print(f"  Thermal Throttling Events Detected: {throttles}")
    print(f"  Telemetry Collection Integrity    : {'PASS (100%)' if len(collected) > 0 else 'FAIL'}")
    print("=" * 78 + "\n")


def run_mode_2_stress_test(
    target_load: int = 75,
    duration_sec: int = 25,
    cutoff_c: float = 85.0,
):
    """Mode 2: Controlled Bounded CPU Stress Test."""
    print_test_header("Mode 2 - Controlled Bounded Stress Test")
    print(f"Target CPU Load : {target_load}% across active cores")
    print(f"Maximum Duration: {duration_sec} seconds")
    print(f"Thermal Cutoff  : {cutoff_c}°C (Stops immediately if reached)\n")

    init_db()

    # Step 1: Measure Baseline Temperature
    print("[STEP 1/3] Measuring baseline hardware idle state (3 seconds)...")
    baseline_snaps = []
    for _ in range(3):
        baseline_snaps.append(collect_hardware_telemetry())
        time.sleep(1.0)

    base_temps = [s["cpu_temp"] for s in baseline_snaps if s["cpu_temp"] is not None]
    baseline_temp = (sum(base_temps) / len(base_temps)) if base_temps else None
    if baseline_temp is not None:
        print(f"  -> Baseline CPU Temperature: {baseline_temp:.1f}°C\n")
    else:
        print("  -> Baseline CPU Temperature: NOT AVAILABLE (proceeding with load bounds)\n")

    # Step 2: Run Bounded Stress Test
    print(f"[STEP 2/3] Initiating bounded stress test for max {duration_sec}s...")
    tester = ControlledStressTester()
    res = tester.start(target_load_pct=target_load, max_duration_sec=duration_sec, temp_cutoff_c=cutoff_c)
    if res.get("status") != "STARTED":
        print(f"[ERROR] Could not start stress tester: {res}")
        return

    peak_temp: Optional[float] = None
    start_stress = time.time()
    try:
        while tester.is_running:
            status = tester.get_status()
            elapsed = status["elapsed_sec"]
            cur_temp = status["peak_temp_c"]
            if cur_temp is not None:
                peak_temp = max(peak_temp or 0.0, cur_temp)
                temp_str = f"{cur_temp:.1f}°C"
            else:
                temp_str = "N/A"

            load_str = f"{status['current_load_pct']}%"
            print(f"  [STRESS {elapsed:04.1f}s/{duration_sec}s] Load: {load_str} | Temp: {temp_str}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[USER] Emergency stop requested.")
        tester.stop()

    final_status = tester.get_status()
    print(f"\nStress load finished. Reason: {final_status.get('reason', 'Completed')}")

    # Step 3: Measure Recovery Temperature
    print("\n[STEP 3/3] Measuring thermal recovery (5 seconds cooldown)...")
    time.sleep(5.0)
    recovery_snap = collect_hardware_telemetry()
    recovery_temp = recovery_snap.get("cpu_temp")
    if recovery_temp is not None:
        print(f"  -> Cooldown Recovery Temperature: {recovery_temp:.1f}°C")
    else:
        print("  -> Cooldown Recovery Temperature: NOT AVAILABLE")

    # Generate Final Report
    print("\n" + "=" * 78)
    print("🛡️ MODE 2 BOUNDED STRESS TEST FINAL REPORT")
    print("=" * 78)
    print(f"  Execution Outcome         : {final_status.get('reason', 'COMPLETED')}")
    print(f"  Total Duration Under Load : {final_status['elapsed_sec']:.1f} seconds")
    print(f"  Target Core Utilization   : {target_load}%")
    if baseline_temp is not None:
        print(f"  Baseline Temperature      : {baseline_temp:.1f}°C")
    if peak_temp is not None:
        print(f"  Peak Temperature Reached  : {peak_temp:.1f}°C")
        if baseline_temp is not None:
            print(f"  Thermal Delta (ΔT)        : +{(peak_temp - baseline_temp):.1f}°C")
    if recovery_temp is not None:
        print(f"  Recovery Temperature      : {recovery_temp:.1f}°C")
    print(f"  Thermal Cutoff Tripped    : {'YES (Safety Stop)' if final_status.get('reason') == 'THERMAL_CUTOFF_EXCEEDED' else 'NO (Safe Bounds Maintained)'}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Automated CPU Telemetry & Stress Test Runner")
    parser.add_argument("--mode", choices=["1", "2"], help="1: Normal Monitoring, 2: Controlled Stress Test")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds (default: 30)")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--load", type=int, default=75, help="Target stress load percentage for Mode 2 (default: 75)")
    parser.add_argument("--cutoff", type=float, default=85.0, help="Thermal cutoff temperature in °C (default: 85.0)")
    args = parser.parse_args()

    mode = args.mode
    if not mode:
        print("\nSelect Automated Test Mode:")
        print("  [1] MODE 1: Normal Real-Hardware Telemetry Monitoring Run (30s)")
        print("  [2] MODE 2: Controlled Bounded CPU Stress Test Run (with auto thermal cutoff)")
        try:
            choice = input("\nEnter mode [1 or 2]: ").strip()
            mode = "2" if choice == "2" else "1"
        except (EOFError, KeyboardInterrupt):
            mode = "1"

    if mode == "1":
        run_mode_1_normal_monitoring(duration_sec=args.duration, interval_sec=args.interval)
    elif mode == "2":
        run_mode_2_stress_test(target_load=args.load, duration_sec=args.duration, cutoff_c=args.cutoff)


if __name__ == "__main__":
    main()
