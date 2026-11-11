import math
from typing import Dict, Any, Tuple
from agent2_core.config import CoolingProfileConfig
from agent2_core.model.cooling_model import CoolingDynamicsModel, FanStateBelief

class FanUtilityEvaluator:
    """
    Utility-Based Agent Subsystem for Fan RPM & Active Cooling:
    Evaluates candidate target RPMs by balancing Cooling Safety, Acoustic Noise (dBA),
    Power Draw (PWM %), and RPM Flapping/Hunting penalties.
    """

    def __init__(self, profile: CoolingProfileConfig):
        self.profile = profile
        self.last_target_rpm: int = 0

    def compute_thermal_utility(self, predicted_temp: float) -> float:
        target = self.profile.target_temp
        critical = self.profile.critical_temp
        silent_thresh = self.profile.silent_temp_threshold

        if predicted_temp <= silent_thresh:
            return 1.0  # Perfect thermal state
        elif predicted_temp <= target:
            return 0.95 - (0.15 * ((predicted_temp - silent_thresh) / (target - silent_thresh)))
        elif predicted_temp < critical:
            ratio = (predicted_temp - target) / (critical - target)
            return 0.80 - (0.60 * ratio)
        else:
            # Critical overheat zone penalty
            overflow = predicted_temp - critical
            clamped_exp = min(50.0, overflow / 2.5)
            penalty = math.exp(clamped_exp) - 1.0
            return max(0.0, 0.20 - penalty)

    def compute_acoustic_noise_utility(self, dba: float) -> float:
        max_dba = self.profile.max_noise_dba
        if self.profile.noise_priority == 0.0:
            return 1.0  # Drone profile: zero noise penalty
        if dba <= 28.0:
            return 1.0  # Whisper quiet
        elif dba <= max_dba:
            return 1.0 - (0.4 * ((dba - 28.0) / (max_dba - 28.0)))
        else:
            overflow = dba - max_dba
            return max(0.0, 0.6 - (0.15 * overflow))

    def compute_power_utility(self, pwm_pct: float) -> float:
        return max(0.0, 1.0 - (pwm_pct / 100.0) * 0.5)

    def evaluate_rpm_setpoint(self, candidate_rpm: int, belief: FanStateBelief, model: CoolingDynamicsModel) -> Tuple[float, Dict[str, float]]:
        # Predict thermal state under candidate RPM
        pred_temp = model.predict_temperature_at_rpm(belief.temperature_c, belief.cpu_percent, candidate_rpm)
        dba = model.compute_acoustic_noise_dba(candidate_rpm)
        pwm_pct = (candidate_rpm / max(1, self.profile.max_rpm)) * 100.0

        u_thermal = self.compute_thermal_utility(pred_temp)
        u_noise = self.compute_acoustic_noise_utility(dba)
        u_power = self.compute_power_utility(pwm_pct)

        w_t = self.profile.thermal_priority
        w_n = self.profile.noise_priority
        w_p = self.profile.power_priority

        w_total = w_t + w_n + w_p
        w_t /= w_total
        w_n /= w_total
        w_p /= w_total

        raw_utility = (w_t * u_thermal) + (w_n * u_noise) + (w_p * u_power)

        # Flapping/Hunting Penalty for small rapid RPM changes
        hunting_penalty = 0.0
        rpm_diff = abs(candidate_rpm - self.last_target_rpm)
        if 50 < rpm_diff < 400 and belief.temperature_c < self.profile.target_temp:
            hunting_penalty = 0.04

        final_utility = max(0.0, round(raw_utility - hunting_penalty, 4))

        breakdown = {
            "u_thermal": round(u_thermal, 4),
            "u_noise": round(u_noise, 4),
            "u_power": round(u_power, 4),
            "predicted_temp": pred_temp,
            "predicted_dba": dba,
            "final_utility": final_utility
        }
        return final_utility, breakdown

    def select_optimal_rpm(self, belief: FanStateBelief, model: CoolingDynamicsModel) -> Tuple[int, float, Dict[str, Any]]:
        # Evaluate candidate RPM setpoints across range
        step = max(200, self.profile.max_rpm // 15)
        candidates = list(range(self.profile.min_rpm, self.profile.max_rpm + 1, step))
        if self.profile.max_rpm not in candidates:
            candidates.append(self.profile.max_rpm)

        best_rpm = self.profile.min_rpm
        best_utility = -1.0
        best_breakdown: Dict[str, float] = {}

        for rpm in candidates:
            u_score, breakdown = self.evaluate_rpm_setpoint(rpm, belief, model)
            if u_score > best_utility:
                best_utility = u_score
                best_rpm = rpm
                best_breakdown = breakdown

        self.last_target_rpm = best_rpm
        return best_rpm, best_utility, {
            "best_target_rpm": best_rpm,
            "best_utility": best_utility,
            "breakdown": best_breakdown
        }
