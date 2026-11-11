"""
Supervisor — top-level orchestrator that wires together ProcessManager,
HealthMonitor, and EventLog into a single control loop.
"""
import time
import threading
import logging
from typing import Dict, List, Optional

from agent5_core.config import SupervisorConfig
from agent5_core.process_manager import ProcessManager, AgentStatus
from agent5_core.health_monitor import HealthMonitor
from agent5_core.event_log import EventLog, EventKind

logger = logging.getLogger("agent5.supervisor")


class Supervisor:
    """
    Agent5 — Meta-Supervisor AI Agent.

    Responsibilities
    ----------------
    1. Spawn Agents 1–4 as child processes.
    2. Monitor liveness via process poll() + HTTP health-checks.
    3. Auto-restart crashed agents within configured retry budget.
    4. Maintain an event log of all state transitions.
    5. Expose aggregated status for CLI and web dashboard consumers.
    """

    def __init__(self, config: SupervisorConfig):
        self.config   = config
        self.event_log = EventLog(max_events=1000)
        self.pm       = ProcessManager(config)
        self._prev_health: Dict[str, bool] = {}
        self._running  = False
        self._loop_thread: Optional[threading.Thread] = None
        self.start_time = time.time()

        # Build health-check URL map
        health_urls = {
            aid: acfg.health_check_url
            for aid, acfg in config.agents.items()
        }

        self.hm = HealthMonitor(
            agents_health_urls=health_urls,
            interval_seconds=config.check_interval_seconds,
            timeout_seconds=3.0,
            on_health_update=self._on_health_update,
        )

    # ─── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start all managed agents and the supervisor loop."""
        self.event_log.log(EventKind.SUPERVISOR, f"{self.config.supervisor_name} starting…")
        logger.info(f"[Supervisor] {self.config.supervisor_name} starting.")

        self.pm.start_all()
        for agent_id in self.config.agents:
            self.event_log.log(EventKind.START, f"Launched {agent_id}", agent_id=agent_id)

        self.hm.start()
        self._running = True

        self._loop_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._loop_thread.start()
        self.event_log.log(EventKind.SUPERVISOR, "All agents started. Supervisor loop active.")

    def stop(self):
        """Gracefully stop all agents and the supervisor."""
        self._running = False
        self.hm.stop()
        self.pm.stop_all()
        for agent_id in self.config.agents:
            self.event_log.log(EventKind.STOPPED, f"Stopped {agent_id}", agent_id=agent_id)
        self.event_log.log(EventKind.SUPERVISOR, "Supervisor shut down.")
        logger.info("[Supervisor] Stopped.")

    # ─── Manual controls (called from web API) ─────────────────────────────────

    def restart_agent(self, agent_id: str) -> bool:
        if agent_id not in self.config.agents:
            return False
        self.event_log.log(EventKind.MANUAL, f"Manual restart of {agent_id}", agent_id=agent_id)
        self.pm.restart_agent(agent_id)
        return True

    def stop_agent(self, agent_id: str) -> bool:
        if agent_id not in self.config.agents:
            return False
        self.event_log.log(EventKind.MANUAL, f"Manual stop of {agent_id}", agent_id=agent_id)
        self.pm.stop_agent(agent_id)
        return True

    # ─── Status aggregation ────────────────────────────────────────────────────

    def get_overview(self) -> dict:
        """Return unified supervisor health overview."""
        agent_statuses = self.pm.get_all_status()
        health_info    = self.hm.get_health()

        # Merge health probe data into each agent record
        for rec in agent_statuses:
            aid = rec["agent_id"]
            h   = health_info.get(aid, {})
            rec["health_ok"]    = h.get("healthy", False)
            rec["latency_ms"]   = h.get("latency_ms", 0.0)

        total    = len(agent_statuses)
        healthy  = sum(1 for r in agent_statuses if r["status"] == AgentStatus.RUNNING.value and r["health_ok"])
        degraded = sum(1 for r in agent_statuses if r["status"] == AgentStatus.DEGRADED.value)
        crashed  = sum(1 for r in agent_statuses if r["status"] in (AgentStatus.CRASHED.value, AgentStatus.STOPPED.value))

        overall = "ALL_HEALTHY" if healthy == total else (
                  "DEGRADED"   if degraded > 0 else
                  "CRITICAL"   if crashed > 0 else "STARTING")

        return {
            "supervisor":    self.config.supervisor_name,
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "overall_status": overall,
            "agents_total":  total,
            "agents_healthy": healthy,
            "agents_degraded": degraded,
            "agents_crashed": crashed,
            "agents":        agent_statuses,
        }

    def get_events(self, limit: int = 100) -> List[dict]:
        return self.event_log.get_all(limit=limit)

    def get_agent_events(self, agent_id: str, limit: int = 50) -> List[dict]:
        return self.event_log.get_for_agent(agent_id, limit=limit)

    # ─── Internal ──────────────────────────────────────────────────────────────

    def _control_loop(self):
        """Main supervisor tick loop — runs every check_interval_seconds."""
        while self._running:
            try:
                self.pm.tick()

                # Log any newly crashed agents
                for rec in self.pm.get_all_status():
                    aid = rec["agent_id"]
                    if rec["status"] == AgentStatus.CRASHED.value:
                        self.event_log.log(
                            EventKind.CRASH,
                            f"{aid} crashed (exit={rec['exit_code']}, restarts={rec['restart_count']})",
                            agent_id=aid,
                        )
                    elif rec["status"] == AgentStatus.RESTARTING.value:
                        self.event_log.log(
                            EventKind.RESTART,
                            f"Auto-restarting {aid} (attempt #{rec['restart_count']})",
                            agent_id=aid,
                        )
            except Exception as exc:
                logger.error(f"[Supervisor] Control loop error: {exc}")

            time.sleep(self.config.check_interval_seconds)

    def _on_health_update(self, agent_id: str, healthy: bool):
        """Callback from HealthMonitor — update process manager and log transitions."""
        self.pm.update_health(agent_id, healthy)

        prev = self._prev_health.get(agent_id)
        if prev is None or prev != healthy:
            kind = EventKind.HEALTH_UP if healthy else EventKind.HEALTH_DN
            msg  = (f"{agent_id} HTTP health-check {'PASS ✅' if healthy else 'FAIL ❌'}")
            self.event_log.log(kind, msg, agent_id=agent_id)
            self._prev_health[agent_id] = healthy
