"""
Event Log — a thread-safe in-memory ring buffer storing supervisor events
(start, crash, restart, health change, manual action).
"""
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class EventKind(str, Enum):
    START      = "START"
    STOPPED    = "STOPPED"
    CRASH      = "CRASH"
    RESTART    = "RESTART"
    HEALTH_UP  = "HEALTH_UP"
    HEALTH_DN  = "HEALTH_DN"
    MANUAL     = "MANUAL"
    SUPERVISOR = "SUPERVISOR"


@dataclass
class SupervisorEvent:
    ts: float
    kind: EventKind
    agent_id: Optional[str]
    message: str

    def to_dict(self) -> dict:
        return {
            "ts":       self.ts,
            "kind":     self.kind.value,
            "agent_id": self.agent_id,
            "message":  self.message,
        }


class EventLog:
    """Thread-safe ring buffer of SupervisorEvents."""

    def __init__(self, max_events: int = 500):
        self._events: List[SupervisorEvent] = []
        self._max = max_events
        self._lock = threading.Lock()

    def log(self, kind: EventKind, message: str, agent_id: Optional[str] = None):
        evt = SupervisorEvent(ts=time.time(), kind=kind, agent_id=agent_id, message=message)
        with self._lock:
            self._events.append(evt)
            if len(self._events) > self._max:
                self._events.pop(0)

    def get_all(self, limit: int = 100) -> List[dict]:
        with self._lock:
            return [e.to_dict() for e in reversed(self._events[-limit:])]

    def get_for_agent(self, agent_id: str, limit: int = 50) -> List[dict]:
        with self._lock:
            return [e.to_dict() for e in reversed(
                [e for e in self._events if e.agent_id == agent_id][-limit:]
            )]
