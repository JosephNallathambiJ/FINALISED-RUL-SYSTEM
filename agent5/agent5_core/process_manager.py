"""
Agent Process Manager — launches, monitors, and restarts sub-agents as OS subprocesses.
"""
import os
import sys
import time
import subprocess
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

from agent5_core.config import ManagedAgentConfig, SupervisorConfig

logger = logging.getLogger("agent5.process_manager")


class AgentStatus(str, Enum):
    STARTING   = "STARTING"
    RUNNING    = "RUNNING"
    DEGRADED   = "DEGRADED"   # process alive but health-check failing
    CRASHED    = "CRASHED"
    RESTARTING = "RESTARTING"
    STOPPED    = "STOPPED"


@dataclass
class AgentRecord:
    cfg: ManagedAgentConfig
    status: AgentStatus = AgentStatus.STOPPED
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    start_time: Optional[float] = None
    restart_count: int = 0
    last_restart_at: Optional[float] = None
    last_seen_alive: Optional[float] = None
    exit_code: Optional[int] = None
    stdout_tail: List[str] = field(default_factory=list)  # last N lines
    stderr_tail: List[str] = field(default_factory=list)
    uptime_seconds: float = 0.0
    health_ok: bool = False

    def to_dict(self) -> dict:
        return {
            "agent_id":      self.cfg.agent_id,
            "name":          self.cfg.name,
            "description":   self.cfg.description,
            "icon":          self.cfg.icon,
            "status":        self.status.value,
            "pid":           self.pid,
            "web_port":      self.cfg.web_port,
            "restart_count": self.restart_count,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "start_time":    self.start_time,
            "exit_code":     self.exit_code,
            "critical":      self.cfg.critical,
            "health_ok":     self.health_ok,
            "stdout_tail":   self.stdout_tail[-20:],
            "stderr_tail":   self.stderr_tail[-20:],
        }


class ProcessManager:
    """
    Manages the full lifecycle of sub-agent processes.
    Spawns, monitors stdout/stderr in background threads, and restarts on crash.
    """

    MAX_TAIL_LINES = 50

    def __init__(self, config: SupervisorConfig):
        self.config = config
        self.records: Dict[str, AgentRecord] = {}
        self._lock = threading.Lock()
        self._reader_threads: Dict[str, List[threading.Thread]] = {}
        self._running = False

        # Resolve paths relative to agent5 root
        self._agent5_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        for agent_id, acfg in config.agents.items():
            self.records[agent_id] = AgentRecord(cfg=acfg)

    # ─── Public API ────────────────────────────────────────────────────────────

    def start_all(self):
        """Launch all configured agents."""
        self._running = True
        for agent_id in self.records:
            self._spawn(agent_id)

    def stop_all(self):
        """Terminate all running agents gracefully."""
        self._running = False
        for agent_id, rec in self.records.items():
            self._terminate(agent_id)

    def restart_agent(self, agent_id: str):
        """Manually restart a specific agent."""
        logger.info(f"[Supervisor] Manual restart requested for {agent_id}")
        self._terminate(agent_id)
        time.sleep(1.0)
        self._spawn(agent_id)

    def stop_agent(self, agent_id: str):
        """Manually stop a specific agent."""
        self._terminate(agent_id)
        with self._lock:
            self.records[agent_id].status = AgentStatus.STOPPED

    def tick(self):
        """
        Health-tick — called by the supervisor loop every check_interval_seconds.
        Detects crashes and auto-restarts agents within retry budget.
        """
        for agent_id, rec in self.records.items():
            proc = rec.process
            if proc is None:
                continue

            ret = proc.poll()
            if ret is not None:
                # Process has exited
                with self._lock:
                    rec.exit_code = ret
                    rec.pid = None
                    rec.process = None
                    rec.status = AgentStatus.CRASHED

                if rec.restart_count < self.config.max_restart_attempts:
                    cooldown = self.config.restart_cooldown_seconds
                    logger.warning(
                        f"[Supervisor] {agent_id} CRASHED (exit={ret}). "
                        f"Restart #{rec.restart_count + 1} in {cooldown}s …"
                    )
                    time.sleep(cooldown)
                    with self._lock:
                        rec.status = AgentStatus.RESTARTING
                    self._spawn(agent_id)
                else:
                    logger.error(
                        f"[Supervisor] {agent_id} exceeded max restarts "
                        f"({self.config.max_restart_attempts}). Giving up."
                    )
            else:
                # Process alive — update uptime
                with self._lock:
                    if rec.start_time:
                        rec.uptime_seconds = time.time() - rec.start_time

    def get_all_status(self) -> List[dict]:
        with self._lock:
            return [rec.to_dict() for rec in self.records.values()]

    def get_agent_status(self, agent_id: str) -> Optional[dict]:
        rec = self.records.get(agent_id)
        return rec.to_dict() if rec else None

    def update_health(self, agent_id: str, healthy: bool):
        """Called by HealthMonitor to update per-agent health-check result."""
        with self._lock:
            rec = self.records.get(agent_id)
            if rec is None:
                return
            rec.health_ok = healthy
            if rec.status == AgentStatus.RUNNING and not healthy:
                rec.status = AgentStatus.DEGRADED
            elif rec.status == AgentStatus.DEGRADED and healthy:
                rec.status = AgentStatus.RUNNING

    # ─── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_path(self, rel_path: str) -> str:
        return os.path.normpath(os.path.join(self._agent5_root, rel_path))

    def _spawn(self, agent_id: str):
        rec = self.records[agent_id]
        script_abs  = self._resolve_path(rec.cfg.script)
        workdir_abs = self._resolve_path(rec.cfg.working_dir)

        cmd = [sys.executable, script_abs] + rec.cfg.args

        logger.info(f"[Supervisor] Spawning {agent_id}: {' '.join(cmd)}")

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=workdir_abs,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            logger.error(f"[Supervisor] Failed to spawn {agent_id}: {exc}")
            with self._lock:
                rec.status = AgentStatus.CRASHED
            return

        with self._lock:
            rec.process      = proc
            rec.pid          = proc.pid
            rec.start_time   = time.time()
            rec.status       = AgentStatus.STARTING
            rec.exit_code    = None
            rec.last_restart_at = time.time()
            rec.restart_count += 1 if rec.restart_count > 0 else 0
            rec.stdout_tail  = []
            rec.stderr_tail  = []

        # Background threads to drain stdout/stderr without blocking
        t_out = threading.Thread(
            target=self._stream_reader,
            args=(proc.stdout, rec.stdout_tail, agent_id, "OUT"),
            daemon=True,
        )
        t_err = threading.Thread(
            target=self._stream_reader,
            args=(proc.stderr, rec.stderr_tail, agent_id, "ERR"),
            daemon=True,
        )
        t_out.start()
        t_err.start()
        self._reader_threads[agent_id] = [t_out, t_err]

        # Mark RUNNING after brief startup delay
        def mark_running():
            time.sleep(2.0)
            with self._lock:
                if rec.status == AgentStatus.STARTING and rec.process and rec.process.poll() is None:
                    rec.status = AgentStatus.RUNNING
                    rec.last_seen_alive = time.time()

        threading.Thread(target=mark_running, daemon=True).start()

    def _terminate(self, agent_id: str):
        rec = self.records[agent_id]
        proc = rec.process
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        with self._lock:
            rec.process = None
            rec.pid = None
            rec.status = AgentStatus.STOPPED

    def _stream_reader(self, stream, tail_list: list, agent_id: str, label: str):
        """Read lines from a subprocess stream and keep a rolling tail."""
        try:
            for line in stream:
                line = line.rstrip("\n")
                with self._lock:
                    tail_list.append(line)
                    if len(tail_list) > self.MAX_TAIL_LINES:
                        tail_list.pop(0)
        except Exception:
            pass
