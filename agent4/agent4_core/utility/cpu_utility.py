from typing import Dict, Any, Tuple
from agent4_core.config import CpuProfileConfig
from agent4_core.model.cpu_model import CpuStateBelief

class CpuUtilityEvaluator:
    def __init__(self, profile: CpuProfileConfig):
        self.profile = profile

    def evaluate_freq_target(self, candidate_freq: float, belief: CpuStateBelief) -> Tuple[float, Dict[str, float]]:
        # Compute performance utility
        ratio = (candidate_freq - self.profile.min_freq_mhz) / max(1, self.profile.max_freq_mhz - self.profile.min_freq_mhz)
        u_perf = min(1.0, max(0.0, ratio))

        # Compute power efficiency utility
        u_power = max(0.0, 1.0 - (ratio ** 1.5))

        w_p = self.profile.perf_priority
        w_pw = self.profile.power_priority
        w_total = w_p + w_pw

        final_u = ((w_p * u_perf) + (w_pw * u_power)) / w_total

        return round(final_u, 4), {
            "u_perf": round(u_perf, 4),
            "u_power": round(u_power, 4),
            "final_utility": round(final_u, 4)
        }

    def select_best_frequency(self, belief: CpuStateBelief) -> Tuple[float, float, Dict[str, Any]]:
        step = (self.profile.max_freq_mhz - self.profile.min_freq_mhz) // 6
        candidates = [self.profile.min_freq_mhz + i * step for i in range(7)]

        best_freq = self.profile.min_freq_mhz
        best_u = -1.0
        best_breakdown = {}

        for freq in candidates:
            u_score, bd = self.evaluate_freq_target(freq, belief)
            if u_score > best_u:
                best_u = u_score
                best_freq = freq
                best_breakdown = bd

        return float(best_freq), best_u, {
            "best_freq_mhz": best_freq,
            "best_utility": best_u,
            "breakdown": best_breakdown
        }
