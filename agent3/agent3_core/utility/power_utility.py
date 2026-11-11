import math
from typing import Dict, Any, Tuple
from agent3_core.config import VoltageProfileConfig
from agent3_core.model.voltage_model import ElectricalDynamicsModel, VoltageStateBelief

class PowerUtilityEvaluator:
    """
    Utility-Based Agent Subsystem for System Voltage & Electrical Power:
    Evaluates candidate regulatory actions balancing Voltage Stability, Brownout Safety,
    Surge Protection, and Energy Conservation.
    """

    def __init__(self, profile: VoltageProfileConfig):
        self.profile = profile

    def compute_stability_utility(self, predicted_vcore: float) -> float:
        nom = self.profile.vcore_nominal
        vmin = self.profile.vcore_min
        vmax = self.profile.vcore_max

        dev_pct = abs(predicted_vcore - nom) / nom * 100.0
        if dev_pct <= self.profile.max_tolerance_pct:
            return 1.0 - (0.05 * (dev_pct / self.profile.max_tolerance_pct))
        elif predicted_vcore < vmin or predicted_vcore > vmax:
            return 0.0  # Hazardous voltage zone
        else:
            return max(0.1, 0.9 - (0.15 * (dev_pct - self.profile.max_tolerance_pct)))

    def compute_safety_utility(self, predicted_vcore: float, battery_v: float) -> float:
        if predicted_vcore < self.profile.vcore_min:
            return 0.05  # Brownout danger
        if predicted_vcore > self.profile.vcore_max:
            return 0.05  # Surge danger
        if battery_v > 0 and battery_v <= self.profile.battery_critical:
            return 0.10  # Low battery flight/system danger
        return 1.0

    def compute_efficiency_utility(self, action: str) -> float:
        action_power_eff = {
            "STABLE_PASSIVE": 1.0,
            "CAP_POWER_DRAW": 0.85,
            "SHED_LOAD_BROWNOUT": 0.70,
            "BOOST_VCORE": 0.50,
            "TRIM_VOLTAGE_SURGE": 0.90,
            "EMERGENCY_VOLTAGE_TRIP": 0.10
        }
        return action_power_eff.get(action, 0.50)

    def evaluate_action(self, action: str, belief: VoltageStateBelief, model: ElectricalDynamicsModel) -> Tuple[float, Dict[str, float]]:
        pred_vcore = model.predict_future_voltage(action, belief.vcore_volts)
        
        u_stab = self.compute_stability_utility(pred_vcore)
        u_safe = self.compute_safety_utility(pred_vcore, belief.battery_volts)
        u_eff = self.compute_efficiency_utility(action)

        w_stab = self.profile.stability_priority
        w_safe = self.profile.safety_priority
        w_eff = self.profile.power_priority

        w_total = w_stab + w_safe + w_eff
        w_stab /= w_total
        w_safe /= w_total
        w_eff /= w_total

        final_u = (w_stab * u_stab) + (w_safe * u_safe) + (w_eff * u_eff)

        breakdown = {
            "u_stability": round(u_stab, 4),
            "u_safety": round(u_safe, 4),
            "u_efficiency": round(u_eff, 4),
            "predicted_vcore": pred_vcore,
            "final_utility": round(final_u, 4)
        }
        return round(final_u, 4), breakdown

    def select_best_power_action(self, belief: VoltageStateBelief, model: ElectricalDynamicsModel) -> Tuple[str, float, Dict[str, Any]]:
        candidate_actions = [
            "STABLE_PASSIVE",
            "BOOST_VCORE",
            "CAP_POWER_DRAW",
            "SHED_LOAD_BROWNOUT",
            "TRIM_VOLTAGE_SURGE",
            "EMERGENCY_VOLTAGE_TRIP"
        ]

        # Prioritize safety emergency actions if brownout or surge detected
        if belief.voltage_status == "BROWNOUT_RISK":
            candidate_actions = ["SHED_LOAD_BROWNOUT", "BOOST_VCORE", "EMERGENCY_VOLTAGE_TRIP"]
        elif belief.voltage_status == "OVER_VOLTAGE_SURGE":
            candidate_actions = ["TRIM_VOLTAGE_SURGE", "CAP_POWER_DRAW", "EMERGENCY_VOLTAGE_TRIP"]

        best_action = candidate_actions[0]
        best_u = -1.0
        best_breakdown: Dict[str, float] = {}

        for act in candidate_actions:
            u_score, breakdown = self.evaluate_action(act, belief, model)
            if u_score > best_u:
                best_u = u_score
                best_action = act
                best_breakdown = breakdown

        return best_action, best_u, {
            "best_action": best_action,
            "best_utility": best_u,
            "breakdown": best_breakdown
        }
