"""Two-computer network connection testing, pairing, and synchronization management."""

from __future__ import annotations

import datetime
import json
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class NetworkManager:
    """Manages connection health, ping tests, and periodic background telemetry synchronization."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("IQAC_API_KEY")
        self.connection_status: str = "DISCONNECTED"  # DISCONNECTED, CONNECTING, ONLINE, OFFLINE, ERROR
        self.remote_host: Optional[str] = None
        self.remote_port: Optional[int] = None
        self.last_sync_time: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_latency_ms: Optional[float] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_sync: threading.Event = threading.Event()
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self.connection_status == "ONLINE"

    def ping_remote(self, host: str, port: int, timeout_sec: float = 3.0) -> Dict[str, Any]:
        """Ping remote device and return connectivity report."""
        res = self.test_connection(host, port, timeout=timeout_sec)
        res["reachable"] = res.get("success", False)
        return res

    def get_connection_status(self) -> Dict[str, Any]:
        return self.get_status()

    def disconnect_peer(self) -> Dict[str, Any]:
        return self.disconnect()

    def test_connection(
        self,
        host: str,
        port: int,
        timeout: float = 3.0,
    ) -> Dict[str, Any]:
        """Test HTTP connectivity and latency to a remote IQAC API server."""
        clean_host = host.strip().replace("http://", "").replace("https://", "").rstrip("/")
        url = f"http://{clean_host}:{port}/"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        start_t = time.time()
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                elapsed_ms = round((time.time() - start_t) * 1000.0, 1)
                data = json.load(resp)
                return {
                    "success": True,
                    "status": "ONLINE",
                    "latency_ms": elapsed_ms,
                    "url": url,
                    "service": data.get("service", "iqac-service"),
                    "hardware": data.get("hardware", {}),
                    "message": f"Successfully connected to {url} in {elapsed_ms}ms",
                }
        except HTTPError as he:
            elapsed_ms = round((time.time() - start_t) * 1000.0, 1)
            msg = f"HTTP {he.code}: {he.reason}"
            if he.code == 401:
                msg = "Authentication failed: Invalid API key"
            return {
                "success": False,
                "status": "ERROR",
                "latency_ms": elapsed_ms,
                "url": url,
                "message": msg,
            }
        except (URLError, TimeoutError, ConnectionRefusedError) as ue:
            elapsed_ms = round((time.time() - start_t) * 1000.0, 1)
            reason = getattr(ue, "reason", str(ue))
            return {
                "success": False,
                "status": "OFFLINE",
                "latency_ms": elapsed_ms,
                "url": url,
                "message": f"Connection failed to {url}: {reason}",
            }
        except Exception as e:
            elapsed_ms = round((time.time() - start_t) * 1000.0, 1)
            return {
                "success": False,
                "status": "ERROR",
                "latency_ms": elapsed_ms,
                "url": url,
                "message": f"Error: {str(e)}",
            }

    def connect(self, host: str, port: int) -> Dict[str, Any]:
        """Pair with remote device and establish periodic synchronization."""
        result = self.test_connection(host, port)
        with self._lock:
            self.remote_host = host.strip()
            self.remote_port = int(port)
            if result["success"]:
                self.connection_status = "ONLINE"
                self.last_latency_ms = result.get("latency_ms")
                self.last_error = None
            else:
                self.connection_status = "OFFLINE"
                self.last_error = result.get("message")
        return result

    def disconnect(self) -> Dict[str, Any]:
        """Stop synchronization and disconnect from remote device."""
        self.stop_background_sync()
        with self._lock:
            self.connection_status = "DISCONNECTED"
            self.remote_host = None
            self.remote_port = None
            self.last_error = None
        return {"status": "DISCONNECTED", "message": "Disconnected from remote peer"}

    def get_status(self) -> Dict[str, Any]:
        """Return current pairing and synchronization status."""
        with self._lock:
            return {
                "status": self.connection_status,
                "remote_host": self.remote_host,
                "remote_port": self.remote_port,
                "last_sync_time": self.last_sync_time,
                "last_latency_ms": self.last_latency_ms,
                "last_error": self.last_error,
            }

    def start_background_sync(
        self,
        monitor_url: str,
        device_id: str,
        interval: float = 10.0,
    ) -> None:
        """Start background polling thread to synchronize records from the monitoring machine."""
        self.stop_background_sync()
        self._stop_sync.clear()

        def _sync_worker():
            import sys
            from pathlib import Path
            pred_path = Path(__file__).resolve().parents[1] / "iqac-prediction"
            if str(pred_path) not in sys.path:
                sys.path.insert(0, str(pred_path))
            from monitor_client import sync_daily_records

            print(f"[NetworkManager] Background sync started for {monitor_url} (device: {device_id})")
            while not self._stop_sync.is_set():
                try:
                    res = sync_daily_records(monitor_url, device_id, timeout=min(interval, 8.0))
                    with self._lock:
                        self.connection_status = "ONLINE"
                        self.last_sync_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        self.last_error = None
                except Exception as exc:
                    with self._lock:
                        self.connection_status = "OFFLINE"
                        self.last_error = f"Sync failed: {exc}"
                self._stop_sync.wait(timeout=max(2.0, interval))

        self._sync_thread = threading.Thread(target=_sync_worker, daemon=True, name="bg-network-sync")
        self._sync_thread.start()

    def stop_background_sync(self) -> None:
        """Signal background synchronization thread to stop."""
        self._stop_sync.set()
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=1.0)
        self._sync_thread = None
