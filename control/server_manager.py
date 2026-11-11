"""Functional remote hosting service and network management engine for the dashboard."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.request

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_BIN = sys.executable

from control.device_manager import get_network_interfaces, get_primary_ip


def is_port_bound(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is currently listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def detect_vpn_and_tailscale() -> Dict[str, Any]:
    """Detect presence of Tailscale, WireGuard, or VPN network interfaces."""
    interfaces = get_network_interfaces()
    tailscale_ip: Optional[str] = None
    vpn_interfaces: List[Dict[str, Any]] = []

    # 1. Scan network interfaces for 100.x (CGNAT / Tailscale) and utun / wg
    for name, iface in interfaces.items():
        ip = iface.get("ip", "")
        # Tailscale assigns 100.64.0.0/10 addresses
        if ip.startswith("100."):
            tailscale_ip = ip
        if name.startswith("utun") or name.startswith("wg") or "vpn" in name.lower() or "tailscale" in name.lower():
            vpn_interfaces.append({"name": name, "ip": ip, "is_up": iface.get("is_up", True)})

    # 2. Check Tailscale CLI availability
    tailscale_cli = shutil.which("tailscale")
    is_tailscale_installed = tailscale_cli is not None

    return {
        "tailscale_installed": is_tailscale_installed,
        "tailscale_ip": tailscale_ip,
        "vpn_active": len(vpn_interfaces) > 0 or tailscale_ip is not None,
        "vpn_interfaces": vpn_interfaces,
        "tailscale_cli_path": tailscale_cli,
    }


class ConnectedClient:
    """Tracks a connected remote client session."""

    def __init__(
        self,
        client_id: str,
        ip_address: str,
        device_name: str = "Remote Host",
        role: str = "VIEWER",
        auth_status: str = "AUTHORIZED",
    ):
        self.client_id = client_id
        self.ip_address = ip_address
        self.device_name = device_name
        self.role = role
        self.auth_status = auth_status
        self.connected_at = time.time()
        self.last_heartbeat = time.time()
        self.sync_state = "IDLE"

    def touch(self) -> None:
        self.last_heartbeat = time.time()

    def to_dict(self) -> Dict[str, Any]:
        duration_sec = int(time.time() - self.connected_at)
        return {
            "client_id": self.client_id,
            "device_name": self.device_name,
            "ip_address": self.ip_address,
            "role": self.role,
            "auth_status": self.auth_status,
            "connected_at_iso": datetime.datetime.fromtimestamp(
                self.connected_at, datetime.timezone.utc
            ).isoformat(),
            "last_heartbeat_iso": datetime.datetime.fromtimestamp(
                self.last_heartbeat, datetime.timezone.utc
            ).isoformat(),
            "connection_duration_sec": duration_sec,
            "sync_state": self.sync_state,
            "is_alive": (time.time() - self.last_heartbeat) < 45.0,
        }


class ServerManager:
    """Manages the lifecycle, network binding, client tracking, and diagnostics of the hosting server."""

    def __init__(self):
        self.status: str = "STOPPED"  # STOPPED, STARTING, RUNNING, ERROR
        self.host: str = "0.0.0.0"     # 0.0.0.0 enables LAN access
        self.port: int = 8080
        self.protocol: str = "http"
        self.server_process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.last_error: Optional[str] = None
        self.clients: Dict[str, ConnectedClient] = {}
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = threading.Event()

    @property
    def is_running(self) -> bool:
        if self.server_process is not None:
            return self.server_process.poll() is None
        # Also verify socket availability if running as external process
        return is_port_bound(self.port, "127.0.0.1")

    @property
    def uptime_seconds(self) -> int:
        if self.start_time and self.is_running:
            return int(time.time() - self.start_time)
        return 0

    def get_access_urls(self) -> Dict[str, str]:
        """Compile exact access URLs for local, LAN, and VPN clients."""
        primary_ip = get_primary_ip()
        vpn_info = detect_vpn_and_tailscale()
        proto = self.protocol

        urls = {
            "local_url": f"{proto}://127.0.0.1:{self.port}",
            "lan_url": f"{proto}://{primary_ip}:{self.port}" if self.host == "0.0.0.0" else "LAN Access Disabled (Binding: 127.0.0.1)",
        }

        if vpn_info["tailscale_ip"] and self.host == "0.0.0.0":
            urls["tailscale_url"] = f"{proto}://{vpn_info['tailscale_ip']}:{self.port}"
        else:
            urls["tailscale_url"] = None

        return urls

    def get_status(self) -> Dict[str, Any]:
        """Return comprehensive server hosting profile and live telemetry."""
        with self._lock:
            # Check process health
            if self.server_process is not None:
                if self.server_process.poll() is not None:
                    self.status = "STOPPED"
                    self.server_process = None
                else:
                    self.status = "RUNNING"
            elif is_port_bound(self.port, "127.0.0.1"):
                self.status = "RUNNING"
            else:
                if self.status != "STARTING":
                    self.status = "STOPPED"

            primary_ip = get_primary_ip()
            vpn_info = detect_vpn_and_tailscale()
            access_urls = self.get_access_urls()

            # Active clients list
            client_list = [c.to_dict() for c in self.clients.values()]

            return {
                "status": self.status,
                "is_running": self.is_running,
                "host": self.host,
                "port": self.port,
                "protocol": self.protocol,
                "hostname": socket.gethostname(),
                "primary_ip": primary_ip,
                "uptime_seconds": self.uptime_seconds,
                "started_at": (
                    datetime.datetime.fromtimestamp(self.start_time, datetime.timezone.utc).isoformat()
                    if self.start_time
                    else None
                ),
                "access_urls": access_urls,
                "connected_clients_count": len([c for c in client_list if c["is_alive"]]),
                "connected_clients": client_list,
                "last_error": self.last_error,
                "vpn_info": vpn_info,
                "lan_accessible": self.host == "0.0.0.0",
                "binding_notice": (
                    f"Bound to {self.host}:{self.port}. Devices on the same Wi-Fi can open {access_urls['lan_url']}."
                    if self.host == "0.0.0.0"
                    else f"Bound to 127.0.0.1:{self.port}. Remote devices cannot connect until LAN hosting is enabled."
                ),
            }

    def start(self, host: str = "0.0.0.0", port: int = 8080) -> Dict[str, Any]:
        """Launch dashboard hosting service on specified host and port."""
        with self._lock:
            if self.is_running:
                return {
                    "status": "ALREADY_RUNNING",
                    "message": f"Server is already running on {self.host}:{self.port}",
                    "access_urls": self.get_access_urls(),
                }

            self.host = host
            self.port = int(port)
            self.status = "STARTING"
            self.last_error = None

            # Check port collision
            if is_port_bound(self.port, "127.0.0.1"):
                self.status = "ERROR"
                self.last_error = f"Port {self.port} is already in use by another application"
                return {"status": "ERROR", "message": self.last_error}

            # Spawn Uvicorn server in a supervised child process
            cmd = [
                PYTHON_BIN,
                "-m",
                "uvicorn",
                "dashboard.app:app",
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--log-level",
                "info",
            ]

            try:
                self.server_process = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                self.start_time = time.time()
                # Give server 1.5 seconds to bind
                time.sleep(1.5)

                if self.server_process.poll() is not None:
                    # Exited prematurely
                    out = self.server_process.stdout.read() if self.server_process.stdout else "Unknown error"
                    self.status = "ERROR"
                    self.last_error = f"Server failed to start: {out[:200]}"
                    self.server_process = None
                    return {"status": "ERROR", "message": self.last_error}

                self.status = "RUNNING"
                return {
                    "status": "RUNNING",
                    "host": self.host,
                    "port": self.port,
                    "access_urls": self.get_access_urls(),
                    "pid": self.server_process.pid,
                    "message": f"Server successfully started on {self.host}:{self.port}",
                }
            except Exception as e:
                self.status = "ERROR"
                self.last_error = str(e)
                return {"status": "ERROR", "message": str(e)}

    def stop(self) -> Dict[str, Any]:
        """Gracefully terminate hosting server process."""
        with self._lock:
            if self.server_process is None:
                self.status = "STOPPED"
                self.start_time = None
                return {"status": "STOPPED", "message": "Server was not running"}

            pid = self.server_process.pid
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=2.0)
            except Exception:
                try:
                    self.server_process.kill()
                except Exception:
                    pass

            self.server_process = None
            self.status = "STOPPED"
            self.start_time = None
            return {"status": "STOPPED", "message": f"Server process {pid} stopped successfully"}

    def restart(self, host: Optional[str] = None, port: Optional[int] = None) -> Dict[str, Any]:
        """Perform zero-hang restart of the hosting service."""
        target_host = host or self.host
        target_port = port or self.port
        self.stop()
        time.sleep(0.5)
        return self.start(host=target_host, port=target_port)

    def test_connection(self, url: Optional[str] = None) -> Dict[str, Any]:
        """Verify reachability of the access URL."""
        target = url or f"{self.protocol}://127.0.0.1:{self.port}/health"
        start_t = time.time()
        try:
            req = urllib.request.Request(target, headers={"User-Agent": "ControlCenter-Diagnose/2.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                latency_ms = round((time.time() - start_t) * 1000, 1)
                return {
                    "reachable": True,
                    "status_code": resp.status,
                    "latency_ms": latency_ms,
                    "target_url": target,
                    "message": f"Successfully reached {target} ({resp.status} OK in {latency_ms}ms)",
                }
        except Exception as e:
            latency_ms = round((time.time() - start_t) * 1000, 1)
            return {
                "reachable": False,
                "latency_ms": latency_ms,
                "target_url": target,
                "error": str(e),
                "message": f"Failed to reach {target}: {e}",
            }

    # Client Registry Methods
    def register_client_heartbeat(
        self,
        client_id: str,
        ip_address: str,
        device_name: str = "Remote Host",
        role: str = "VIEWER",
    ) -> None:
        """Record an active client connection."""
        with self._lock:
            if client_id not in self.clients:
                self.clients[client_id] = ConnectedClient(
                    client_id=client_id,
                    ip_address=ip_address,
                    device_name=device_name,
                    role=role,
                )
            else:
                self.clients[client_id].touch()
                self.clients[client_id].ip_address = ip_address

    def disconnect_client(self, client_id: str) -> bool:
        """Remove a client from active connection registry."""
        with self._lock:
            if client_id in self.clients:
                del self.clients[client_id]
                return True
            return False


server_mgr = ServerManager()

