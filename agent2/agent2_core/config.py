import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class CoolingProfileConfig:
    name: str
    min_rpm: int
    max_rpm: int
    silent_temp_threshold: float
    target_temp: float
    critical_temp: float
    max_noise_dba: float
    thermal_priority: float
    noise_priority: float
    power_priority: float
    kp: float
    ki: float
    kd: float

@dataclass
class Agent2Config:
    active_profile: str
    db_path: str
    agent_interval_seconds: float
    profiles: Dict[str, CoolingProfileConfig]

    @property
    def current_profile(self) -> CoolingProfileConfig:
        if self.active_profile in self.profiles:
            return self.profiles[self.active_profile]
        return list(self.profiles.values())[0]

def load_config(config_path: str = "config.yaml") -> Agent2Config:
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    profiles_dict = {}
    for p_key, p_val in data.get("profiles", {}).items():
        profiles_dict[p_key] = CoolingProfileConfig(
            name=p_val.get("name", p_key),
            min_rpm=int(p_val.get("min_rpm", 0)),
            max_rpm=int(p_val.get("max_rpm", 5000)),
            silent_temp_threshold=float(p_val.get("silent_temp_threshold", 48.0)),
            target_temp=float(p_val.get("target_temp", 65.0)),
            critical_temp=float(p_val.get("critical_temp", 85.0)),
            max_noise_dba=float(p_val.get("max_noise_dba", 45.0)),
            thermal_priority=float(p_val.get("thermal_priority", 0.8)),
            noise_priority=float(p_val.get("noise_priority", 0.5)),
            power_priority=float(p_val.get("power_priority", 0.4)),
            kp=float(p_val.get("kp", 2.5)),
            ki=float(p_val.get("ki", 0.15)),
            kd=float(p_val.get("kd", 0.80))
        )

    return Agent2Config(
        active_profile=data.get("active_profile", "laptop"),
        db_path=data.get("db_path", "rpm_cooling_history.db"),
        agent_interval_seconds=float(data.get("agent_interval_seconds", 1.5)),
        profiles=profiles_dict
    )
