"""Controlled CPU stress testing engine with thermal safety cutoffs and immediate stop."""

from __future__ import annotations

import math
import multiprocessing
import threading
import time
from typing import Any, Dict, List, Optional

import psutil

from control.measurement_manager import read_real_cpu_temperature


def _worker_stress_slice(stop_event: multiprocessing.Event, duty_cycle: float):
    """Work-sleep duty cycle worker generating target CPU utilization on one core."""
    period = 0.05
    work_time = period * max(0.1, min(0.95, duty_cycle))
    sleep_time = period - work_time

    while not stop_event.is_set():
        # Active compute slice
        t_start = time.time()
        while time.time() - t_start < work_time:
            _ = math.sqrt(1234567.89) * math.sin(0.98765)
        # Inactive cooldown slice
        if sleep_time > 0:
            time.sleep(sleep_time)


class ControlledStressTester:
    """Supervises a bounded, safe CPU workload with thermal auto-cutoff."""

    def __init__(self):
        self.is_running: bool = False
        self.target_load_pct: float = 60.0
        self.max_duration_sec: float = 30.0
        self.temp_cutoff_c: float = 85.0
        self.start_time: Optional[float] = None
        self.stop_reason: Optional[str] = None
        self.last_elapsed: float = 0.0
        self._stop_event: Optional[multiprocessing.Event] = None
        self._workers: List[multiprocessing.Process] = []
        self._watcher_thread: Optional[threading.Thread] = None

    def start(
        self,
        target_load_pct: float = 60.0,
        max_duration_sec: float = 30.0,
        temp_cutoff_c: float = 85.0,
    ) -> Dict[str, Any]:
        """Start bounded stress test with thermal guardian watcher."""
        if self.is_running:
            return {"status": "ALREADY_RUNNING"}

        self.target_load_pct = max(10.0, min(95.0, float(target_load_pct)))
        self.max_duration_sec = max(5.0, min(300.0, float(max_duration_sec)))
        self.temp_cutoff_c = max(60.0, min(100.0, float(temp_cutoff_c)))
        self.stop_reason = None
        self.start_time = time.time()

        self._stop_event = multiprocessing.Event()
        self._workers = []

        # Spawn workers for physical cores
        num_cores = max(1, psutil.cpu_count(logical=False) or 2)
        duty = self.target_load_pct / 100.0

        for _ in range(num_cores):
            p = multiprocessing.Process(
                target=_worker_stress_slice,
                args=(self._stop_event, duty),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        self.is_running = True

        # Launch watcher thread to enforce max duration and thermal cutoff
        self._watcher_thread = threading.Thread(target=self._watchdog, daemon=True, name="stress-watchdog")
        self._watcher_thread.start()

        return {
            "status": "STARTED",
            "is_running": True,
            "cores_active": num_cores,
            "target_load_pct": self.target_load_pct,
            "max_duration_sec": self.max_duration_sec,
            "temp_cutoff_c": self.temp_cutoff_c,
        }

    def _watchdog(self) -> None:
        """Continuously inspect elapsed time and temperature; stop workers on threshold breach."""
        while self.is_running:
            now = time.time()
            elapsed = now - (self.start_time or now)

            # Check time limit
            if elapsed >= self.max_duration_sec:
                self.stop(reason="DURATION_LIMIT_REACHED")
                break

            # Check thermal cutoff
            temp = read_real_cpu_temperature()
            if temp is not None and temp >= self.temp_cutoff_c:
                self.stop(reason=f"THERMAL_CUTOFF_REACHED ({temp:.1f}°C >= {self.temp_cutoff_c}°C)")
                break

            time.sleep(0.5)

    def stop(self, reason: str = "MANUAL_STOP") -> Dict[str, Any]:
        """Stop all stress workers immediately."""
        if not self.is_running:
            return {"status": "NOT_RUNNING"}

        self.is_running = False
        self.stop_reason = reason

        if self._stop_event:
            self._stop_event.set()

        for p in self._workers:
            try:
                p.terminate()
                p.join(timeout=0.5)
            except Exception:
                pass
        self._workers.clear()

        self.last_elapsed = time.time() - (self.start_time or time.time())
        return {
            "status": "STOPPED",
            "reason": reason,
            "elapsed_seconds": round(self.last_elapsed, 1),
        }

    def get_status(self) -> Dict[str, Any]:
        """Return the current stress test status."""
        elapsed = (time.time() - self.start_time) if (self.is_running and self.start_time) else self.last_elapsed
        cur_temp = read_real_cpu_temperature()
        return {
            "active": self.is_running,
            "is_running": self.is_running,
            "target_load_pct": self.target_load_pct,
            "current_load_pct": self.target_load_pct if self.is_running else 0.0,
            "max_duration_sec": self.max_duration_sec,
            "temp_cutoff_c": self.temp_cutoff_c,
            "elapsed_sec": round(elapsed, 1),
            "elapsed_seconds": round(elapsed, 1),
            "peak_temp_c": cur_temp,
            "stop_reason": self.stop_reason,
            "reason": self.stop_reason or ("RUNNING" if self.is_running else "READY"),
        }
