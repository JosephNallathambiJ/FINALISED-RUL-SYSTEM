"""Configuration loader for Agent5 Supervisor."""
import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ManagedAgentConfig:
    agent_id: str
    name: str
    description: str
    icon: str
    script: str
    working_dir: str
    args: List[str]
    web_port: int
    health_check_url: str
    critical: bool


@dataclass
class SupervisorConfig:
    supervisor_name: str
    check_interval_seconds: float
    restart_cooldown_seconds: float
    max_restart_attempts: int
    web_port: int
    web_host: str
    agents: Dict[str, ManagedAgentConfig] = field(default_factory=dict)


def load_config(config_path: str = "config.yaml") -> SupervisorConfig:
    """Load and parse supervisor configuration."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
    
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    agents = {}
    for agent_id, acfg in raw.get("agents", {}).items():
        agents[agent_id] = ManagedAgentConfig(
            agent_id=agent_id,
            name=acfg.get("name", agent_id),
            description=acfg.get("description", ""),
            icon=acfg.get("icon", "🤖"),
            script=acfg.get("script", f"../{agent_id}/main.py"),
            working_dir=acfg.get("working_dir", f"../{agent_id}"),
            args=acfg.get("args", []),
            web_port=acfg.get("web_port", 8000),
            health_check_url=acfg.get("health_check_url", ""),
            critical=acfg.get("critical", True),
        )

    return SupervisorConfig(
        supervisor_name=raw.get("supervisor_name", "Supervisor"),
        check_interval_seconds=float(raw.get("check_interval_seconds", 3.0)),
        restart_cooldown_seconds=float(raw.get("restart_cooldown_seconds", 5.0)),
        max_restart_attempts=int(raw.get("max_restart_attempts", 5)),
        web_port=int(raw.get("web_port", 8005)),
        web_host=raw.get("web_host", "127.0.0.1"),
        agents=agents,
    )
