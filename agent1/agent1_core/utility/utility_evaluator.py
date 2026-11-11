import math
from typing import Dict, Any, List, Tuple
from agent1_core.config import ProfileConfig
from agent1_core.model.world_model import WorldModel, SystemBeliefState

class UtilityEvaluator:
    """
    Utility-Based Agent Subsystem:
    Evaluates candidate actions using multi-attribute utility curves balancing
    Thermal Safety, System Compute Performance, Energy Conservation, and Action Stability.
    """

    def __init__(self, profile: ProfileConfig):
        self.profile = profile
        self.last_chosen_action: str = "PASSIVE"

    def compute_thermal_utility(self, predicted_temp: float) -> float:
        """
        Non-linear sigmoidal/exponential utility curve for temperature safety.
        Returns score from 0.0 (catastrophic overheat) to 1.0 (optimal thermal state).
        """
        warning = self.profile.thermal_warning_temp
        critical = self.profile.thermal_critical_temp
        max_safe = self.profile.max_safe_temp

        if predicted_temp <= warning:
            # Safe zone: high linear utility
            return 1.0 - (0.2 * (predicted_temp / warning))
        elif predicted_temp < critical:
            # Warning zone: decreasing utility
            ratio = (predicted_temp - warning) / (critical - warning)
            return 0.8 - (0.5 * ratio)
        else:
            # Critical zone: steep penalty curve towards 0
            overflow = predicted_temp - critical
            clamped_exp = min(50.0, overflow / 3.0)
            penalty = math.exp(clamped_exp) - 1.0
            return max(0.0, 0.3 - penalty)

    def compute_performance_utility(self, action: str) -> float:
        """Utility score for compute throughput preservation under given action."""
        action_perf_scores = {
            "PASSIVE": 1.0,
            "COOL_MODERATE": 0.95,
            "THROTTLE_LIGHT": 0.75,
            "REDUCE_PAYLOAD": 0.65,
            "THROTTLE_HEAVY": 0.40,
            "EMERGENCY_COOLING": 0.20,
            "EMERGENCY_LANDING_ALERT": 0.10,
            "SLEEP_DUTY_CYCLE": 0.05
        }
        return action_perf_scores.get(action, 0.5)

    def compute_power_utility(self, action: str, battery_pct: float, is_charging: bool) -> float:
        """Utility score for power efficiency."""
        if is_charging:
            # AC Power: power constraint is relaxed
            return 0.9
        
        # Battery active: penalize power-hungry actions if battery low
        action_power_cost = {
            "PASSIVE": 0.1,
            "COOL_MODERATE": 0.3,
            "THROTTLE_LIGHT": 0.4,
            "REDUCE_PAYLOAD": 0.5,
            "THROTTLE_HEAVY": 0.7,
            "EMERGENCY_COOLING": 0.95,
            "EMERGENCY_LANDING_ALERT": 0.8,
            "SLEEP_DUTY_CYCLE": 0.05
        }
        cost = action_power_cost.get(action, 0.5)
        # Higher score means better power preservation
        if battery_pct < 20.0:
            # Critical battery: heavy penalty for high power consumption
            return 1.0 - (cost * 1.5)
        return max(0.0, 1.0 - cost)

    def evaluate_action(self, action: str, state: SystemBeliefState, world_model: WorldModel) -> Tuple[float, Dict[str, float]]:
        """
        Computes Total Utility U(S, A) = w_thermal * U_thermal + w_perf * U_perf + w_power * U_power - switching_penalty
        """
        # Model-based forecast
        forecast = world_model.predict_future_state(action, horizon_sec=4.0)
        pred_temp = forecast["predicted_temp_c"]

        u_thermal = self.compute_thermal_utility(pred_temp)
        u_perf = self.compute_performance_utility(action)
        u_power = self.compute_power_utility(action, state.battery_percent, state.is_charging)

        w_thermal = self.profile.safety_priority
        w_perf = self.profile.perf_priority
        w_power = self.profile.power_priority

        # Normalize weights
        w_total = w_thermal + w_perf + w_power
        w_thermal /= w_total
        w_perf /= w_total
        w_power /= w_total

        raw_utility = (w_thermal * u_thermal) + (w_perf * u_perf) + (w_power * u_power)

        # Action inertia / switching stability penalty to avoid flapping
        switching_penalty = 0.0
        if action != self.last_chosen_action and self.last_chosen_action != "PASSIVE":
            switching_penalty = 0.03

        final_utility = max(0.0, round(raw_utility - switching_penalty, 4))

        breakdown = {
            "u_thermal": round(u_thermal, 4),
            "u_perf": round(u_perf, 4),
            "u_power": round(u_power, 4),
            "predicted_temp": pred_temp,
            "final_utility": final_utility
        }
        return final_utility, breakdown

    def select_best_action(self, state: SystemBeliefState, world_model: WorldModel) -> Tuple[str, float, Dict[str, Any]]:
        """Selects action A* = argmax_A U(S, A)."""
        candidate_actions = self.profile.default_actions
        best_action = "PASSIVE"
        best_utility = -1.0
        best_breakdown: Dict[str, float] = {}
        all_evaluations: Dict[str, Any] = {}

        for act in candidate_actions:
            utility_score, breakdown = self.evaluate_action(act, state, world_model)
            all_evaluations[act] = breakdown
            if utility_score > best_utility:
                best_utility = utility_score
                best_action = act
                best_breakdown = breakdown

        self.last_chosen_action = best_action
        return best_action, best_utility, {
            "best_action": best_action,
            "best_utility": best_utility,
            "selected_breakdown": best_breakdown,
            "all_candidates": all_evaluations
        }
