import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CpuProfileConfig:
    name: str
    min_freq_mhz: int
    max_freq_mhz: int
    target_cpu_load: float
    governor_mode: str
    perf_priority: float
    power_priority: float

@dataclass
class Agent4Config:
    active_profile: str
    db_path: str
    agent_interval_seconds: float
    profiles: Dict[str, CpuProfileConfig]

    @property
    def current_profile(self) -> CpuProfileConfig:
        if self.active_profile in self.profiles:
            return self.profiles[self.active_profile]
        return list(self.profiles.values())[0]

def load_config(config_path: str = "config.yaml") -> Agent4Config:
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    profiles_dict = {}
    for p_key, p_val in data.get("profiles", {}).items():
        profiles_dict[p_key] = CpuProfileConfig(
            name=p_val.get("name", p_key),
            min_freq_mhz=int(p_val.get("min_freq_mhz", 800)),
            max_freq_mhz=int(p_val.get("max_freq_mhz", 3800)),
            target_cpu_load=float(p_val.get("target_cpu_load", 65.0)),
            governor_mode=p_val.get("governor_mode", "POWERSAVE"),
            perf_priority=float(p_val.get("perf_priority", 0.7)),
            power_priority=float(p_val.get("power_priority", 0.75))
        )

    return Agent4Config(
        active_profile=data.get("active_profile", "laptop"),
        db_path=data.get("db_path", "cpu_freq_history.db"),
        agent_interval_seconds=float(data.get("agent_interval_seconds", 1.5)),
        profiles=profiles_dict
    )
