"""
Health Monitor — polls each sub-agent's HTTP health endpoint and reports
availability back to the ProcessManager.
"""
import time
import threading
import logging
from typing import Dict, Callable, Optional
from urllib import request as urllib_request
from urllib.error import URLError

logger = logging.getLogger("agent5.health_monitor")


class HealthMonitor:
    """
    Periodically polls each agent's /api/status HTTP endpoint.
    Reports health results via callback to the ProcessManager.
    """

    def __init__(
        self,
        agents_health_urls: Dict[str, str],
        interval_seconds: float = 5.0,
        timeout_seconds: float = 3.0,
        on_health_update: Optional[Callable[[str, bool], None]] = None,
    ):
        self.agents_health_urls = agents_health_urls
        self.interval_seconds   = interval_seconds
        self.timeout_seconds    = timeout_seconds
        self.on_health_update   = on_health_update

        self._health_map: Dict[str, bool]  = {aid: False for aid in agents_health_urls}
        self._latency_map: Dict[str, float] = {aid: 0.0 for aid in agents_health_urls}
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ─── Public ────────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("[HealthMonitor] Started polling loop.")

    def stop(self):
        self._running = False

    def get_health(self) -> Dict[str, dict]:
        with self._lock:
            return {
                aid: {
                    "healthy": self._health_map.get(aid, False),
                    "latency_ms": round(self._latency_map.get(aid, 0.0) * 1000, 1),
                }
                for aid in self.agents_health_urls
            }

    def is_healthy(self, agent_id: str) -> bool:
        with self._lock:
            return self._health_map.get(agent_id, False)

    # ─── Internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        while self._running:
            for agent_id, url in self.agents_health_urls.items():
                healthy, latency = self._probe(url)
                with self._lock:
                    self._health_map[agent_id]   = healthy
                    self._latency_map[agent_id]  = latency
                if self.on_health_update:
                    try:
                        self.on_health_update(agent_id, healthy)
                    except Exception as exc:
                        logger.debug(f"[HealthMonitor] callback error for {agent_id}: {exc}")
            time.sleep(self.interval_seconds)

    def _probe(self, url: str):
        """Return (healthy, latency_seconds). Healthy = HTTP 200 within timeout."""
        if not url:
            return False, 0.0
        t0 = time.perf_counter()
        try:
            with urllib_request.urlopen(url, timeout=self.timeout_seconds) as resp:
                latency = time.perf_counter() - t0
                return resp.status == 200, latency
        except (URLError, OSError, Exception):
            latency = time.perf_counter() - t0
            return False, latency
