import time
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class VoltageStateBelief:
    timestamp: float
    vcore_volts: float
    v12_volts: float
    v5_volts: float
    v33_volts: float
    battery_volts: float
    vcore_velocity_v_per_sec: float
    power_watts: float
    voltage_status: str  # "NORMAL", "UNDER_VOLTAGE_SAG", "OVER_VOLTAGE_SURGE", "BROWNOUT_RISK", "CRITICAL_BATTERY"
    tolerance_deviation_pct: float

class ElectricalDynamicsModel:
    """
    Model-Based Agent Subsystem for Voltage & Power Dynamics:
    1. Tracks current belief state of multi-rail system voltages
    2. Evaluates voltage tolerance deviations (±5% / ±10%)
    3. Predicts expected next voltage states under power regulatory actions.
    """

    def __init__(self, vcore_nom: float = 1.15, vcore_min: float = 0.85, vcore_max: float = 1.35, battery_crit: float = 9.9):
        self.vcore_nom = vcore_nom
        self.vcore_min = vcore_min
        self.vcore_max = vcore_max
        self.battery_crit = battery_crit
        self.last_vcore = vcore_nom
        self.last_time = time.time()

    def update_belief_state(self, rails: Dict[str, float], cpu_p: float) -> VoltageStateBelief:
        now = time.time()
        dt = max(0.1, now - self.last_time)
        
        vcore = rails.get("vcore_volts", self.vcore_nom)
        v12 = rails.get("v12_volts", 12.0)
        v5 = rails.get("v5_volts", 5.0)
        v33 = rails.get("v33_volts", 3.3)
        bat_v = rails.get("battery_volts", 0.0)
        watts = rails.get("power_watts", round(vcore * (15.0 + (cpu_p * 0.45)), 1))

        # Calculate voltage velocity (dV/dt)
        velocity = (vcore - self.last_vcore) / dt
        self.last_vcore = vcore
        self.last_time = now

        # Calculate percentage deviation from nominal
        deviation_pct = abs(vcore - self.vcore_nom) / self.vcore_nom * 100.0

        # Health status evaluation
        if vcore < self.vcore_min:
            status = "BROWNOUT_RISK"
        elif vcore > self.vcore_max:
            status = "OVER_VOLTAGE_SURGE"
        elif bat_v > 0 and bat_v <= self.battery_crit:
            status = "CRITICAL_BATTERY"
        elif deviation_pct > 6.0 and vcore < self.vcore_nom:
            status = "UNDER_VOLTAGE_SAG"
        else:
            status = "NORMAL"

        return VoltageStateBelief(
            timestamp=now,
            vcore_volts=vcore,
            v12_volts=v12,
            v5_volts=v5,
            v33_volts=v33,
            battery_volts=bat_v,
            vcore_velocity_v_per_sec=round(velocity, 4),
            power_watts=watts,
            voltage_status=status,
            tolerance_deviation_pct=round(deviation_pct, 2)
        )

    def predict_future_voltage(self, candidate_action: str, current_vcore: float, horizon_sec: float = 2.0) -> float:
        """Forecasts expected VCore under power regulatory actions."""
        adj = 0.0
        if candidate_action == "BOOST_VCORE":
            adj = +0.03
        elif candidate_action == "SHED_LOAD_BROWNOUT":
            adj = +0.05  # Reduced load lifts voltage back up
        elif candidate_action == "CAP_POWER_DRAW":
            adj = +0.02
        elif candidate_action == "TRIM_VOLTAGE_SURGE":
            adj = -0.04

        pred = current_vcore + adj
        return round(pred, 3)
