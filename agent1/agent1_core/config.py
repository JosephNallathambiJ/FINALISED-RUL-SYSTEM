import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class ProfileConfig:
    name: str
    thermal_warning_temp: float
    thermal_critical_temp: float
    max_safe_temp: float
    power_priority: float
    perf_priority: float
    safety_priority: float
    cooling_capacity: float
    default_actions: List[str]

@dataclass
class AgentConfig:
    active_profile: str
    db_path: str
    agent_interval_seconds: float
    profiles: Dict[str, ProfileConfig]

    @property
    def current_profile(self) -> ProfileConfig:
        if self.active_profile in self.profiles:
            return self.profiles[self.active_profile]
        # fallback to laptop
        return list(self.profiles.values())[0]

def load_config(config_path: str = "config.yaml") -> AgentConfig:
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    profiles_dict = {}
    for p_key, p_val in data.get("profiles", {}).items():
        profiles_dict[p_key] = ProfileConfig(
            name=p_val.get("name", p_key),
            thermal_warning_temp=float(p_val.get("thermal_warning_temp", 70.0)),
            thermal_critical_temp=float(p_val.get("thermal_critical_temp", 85.0)),
            max_safe_temp=float(p_val.get("max_safe_temp", 95.0)),
            power_priority=float(p_val.get("power_priority", 0.5)),
            perf_priority=float(p_val.get("perf_priority", 0.7)),
            safety_priority=float(p_val.get("safety_priority", 0.8)),
            cooling_capacity=float(p_val.get("cooling_capacity", 0.5)),
            default_actions=p_val.get("default_actions", ["PASSIVE", "THROTTLE_LIGHT"])
        )

    return AgentConfig(
        active_profile=data.get("active_profile", "laptop"),
        db_path=data.get("db_path", "temperature_history.db"),
        agent_interval_seconds=float(data.get("agent_interval_seconds", 2.0)),
        profiles=profiles_dict
    )
