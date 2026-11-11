import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class VoltageProfileConfig:
    name: str
    vcore_nominal: float
    vcore_min: float
    vcore_max: float
    v12_nominal: float
    v5_nominal: float
    v33_nominal: float
    battery_nominal: float
    battery_critical: float
    max_tolerance_pct: float
    power_priority: float
    stability_priority: float
    safety_priority: float

@dataclass
class Agent3Config:
    active_profile: str
    db_path: str
    agent_interval_seconds: float
    profiles: Dict[str, VoltageProfileConfig]

    @property
    def current_profile(self) -> VoltageProfileConfig:
        if self.active_profile in self.profiles:
            return self.profiles[self.active_profile]
        return list(self.profiles.values())[0]

def load_config(config_path: str = "config.yaml") -> Agent3Config:
    if not os.path.isabs(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    profiles_dict = {}
    for p_key, p_val in data.get("profiles", {}).items():
        profiles_dict[p_key] = VoltageProfileConfig(
            name=p_val.get("name", p_key),
            vcore_nominal=float(p_val.get("vcore_nominal", 1.15)),
            vcore_min=float(p_val.get("vcore_min", 0.85)),
            vcore_max=float(p_val.get("vcore_max", 1.35)),
            v12_nominal=float(p_val.get("v12_nominal", 12.00)),
            v5_nominal=float(p_val.get("v5_nominal", 5.00)),
            v33_nominal=float(p_val.get("v33_nominal", 3.30)),
            battery_nominal=float(p_val.get("battery_nominal", 11.40)),
            battery_critical=float(p_val.get("battery_critical", 9.90)),
            max_tolerance_pct=float(p_val.get("max_tolerance_pct", 5.0)),
            power_priority=float(p_val.get("power_priority", 0.8)),
            stability_priority=float(p_val.get("stability_priority", 0.9)),
            safety_priority=float(p_val.get("safety_priority", 0.95))
        )

    return Agent3Config(
        active_profile=data.get("active_profile", "laptop"),
        db_path=data.get("db_path", "voltage_history.db"),
        agent_interval_seconds=float(data.get("agent_interval_seconds", 1.5)),
        profiles=profiles_dict
    )
