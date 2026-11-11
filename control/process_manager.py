"""Subprocess manager for background service processes, port checking, and logging."""

from __future__ import annotations

import atexit
import collections
import datetime
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_EXEC = sys.executable


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check whether a local TCP port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def find_available_port(start_port: int, max_attempts: int = 20) -> int:
    """Find the first available port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        if not is_port_in_use(p):
            return p
    return start_port


class LogBuffer:
    """Thread-safe rotating log line buffer for application event streaming."""

    def __init__(self, maxlen: int = 500, max_lines: Optional[int] = None):
        capacity = max_lines if max_lines is not None else maxlen
        self.buffer: Deque[Dict[str, Any]] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, source: str, message: str, level: str = "INFO") -> None:
        with self._lock:
            self.buffer.append({
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "source": source,
                "level": level,
                "message": message.rstrip("\n"),
            })

    def get_logs(self, limit: Optional[int] = None, level: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            logs = list(self.buffer)
            if level and level.upper() != "ALL":
                logs = [l for l in logs if l["level"].upper() == level.upper()]
            if limit and limit > 0:
                return logs[-limit:]
            return logs


class ServiceProcess:
    """Represents a supervised child service process."""

    def __init__(self, name: str, command: List[str], cwd: Path, log_buffer: LogBuffer):
        self.name = name
        self.command = command
        self.cwd = cwd
        self.log_buffer = log_buffer
        self.process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        if self.is_running():
            return True

        self.log_buffer.append("Supervisor", f"Starting {self.name}: {' '.join(self.command)}")
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._reader_thread = threading.Thread(
                target=self._stream_output,
                daemon=True,
                name=f"log-{self.name}",
            )
            self._reader_thread.start()
            return True
        except Exception as e:
            self.log_buffer.append("Supervisor", f"Failed to start {self.name}: {e}", level="ERROR")
            return False

    def _stream_output(self) -> None:
        if not self.process or not self.process.stdout:
            return
        try:
            for line in iter(self.process.stdout.readline, ""):
                if line:
                    self.log_buffer.append(self.name, line)
        except Exception:
            pass

    def stop(self, timeout: float = 3.0) -> None:
        if not self.process:
            return
        self.log_buffer.append("Supervisor", f"Stopping service: {self.name}")
        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, Exception):
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
        self.process = None

    def is_running(self) -> bool:
        if self.process is None:
            return False
        return self.process.poll() is None


class ProcessManager:
    """Coordinates and supervises all child processes of the control center."""

    def __init__(self):
        self.log_buffer = LogBuffer()
        self.services: Dict[str, ServiceProcess] = {}
        self._lock = threading.Lock()
        atexit.register(self.shutdown_all)

    def start_service(self, name: str, script_path: str, args: Optional[List[str]] = None, cwd: Optional[Path] = None) -> bool:
        """Start a managed service if not already running."""
        with self._lock:
            if name in self.services and self.services[name].is_running():
                return True

            work_dir = cwd or ROOT_DIR
            full_cmd = [PYTHON_EXEC, script_path] + (args or [])
            svc = ServiceProcess(name=name, command=full_cmd, cwd=work_dir, log_buffer=self.log_buffer)
            if svc.start():
                self.services[name] = svc
                return True
            return False

    def stop_service(self, name: str) -> bool:
        """Stop a specific managed service."""
        with self._lock:
            if name in self.services:
                self.services[name].stop()
                del self.services[name]
                return True
            return False

    def get_service_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Return running state, PID, and name for all managed services."""
        with self._lock:
            statuses = {}
            for name, svc in self.services.items():
                running = svc.is_running()
                statuses[name] = {
                    "running": running,
                    "pid": svc.process.pid if running and svc.process else None,
                    "command": " ".join(svc.command),
                }
            return statuses

    def is_port_in_use(self, port: int, host: str = "127.0.0.1") -> bool:
        """Check whether a port is currently in use."""
        return is_port_in_use(port, host)

    def start_process(self, name: str, command: List[str], cwd: Optional[Path] = None) -> bool:
        """Start a managed child process directly from a command list."""
        with self._lock:
            if name in self.services and self.services[name].is_running():
                return True
            work_dir = cwd or ROOT_DIR
            svc = ServiceProcess(name=name, command=command, cwd=work_dir, log_buffer=self.log_buffer)
            if svc.start():
                self.services[name] = svc
                return True
            return False

    def stop_all(self) -> None:
        """Convenience alias for shutting down all processes."""
        self.shutdown_all()

    def shutdown_all(self) -> None:
        """Gracefully terminate all running child services."""
        with self._lock:
            for name, svc in list(self.services.items()):
                try:
                    svc.stop(timeout=1.5)
                except Exception:
                    pass
            self.services.clear()
