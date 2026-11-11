from typing import Dict, Any
from agent2_core.learning.rpm_history_store import RpmHistoryStore

class FanLearner:
    """
    Learning Agent Subsystem for Active Cooling:
    Monitors past RPM and temperature history to detect fan degradation,
    bearing friction, or mechanical stall conditions.
    """

    def __init__(self, history_store: RpmHistoryStore):
        self.history_store = history_store
        self.stall_events_detected: int = 0
        self.learned_cooling_efficiency: float = 1.0  # Normalized cooling multiplier

    def update_learning_model(self) -> Dict[str, Any]:
        history = self.history_store.get_recent_history(limit=100)
        if len(history) < 10:
            return {"status": "COLLECTING_DATA", "efficiency": 1.0}

        # Check for PWM > 25% but 0 RPM
        stall_count = sum(1 for r in history if r["target_pwm_pct"] > 25.0 and r["current_rpm"] < 50)
        if stall_count > 3:
            self.stall_events_detected += 1
            self.learned_cooling_efficiency = max(0.2, self.learned_cooling_efficiency - 0.05)

        return {
            "status": "HEALTHY" if stall_count == 0 else "WARNING_STALL_DEGRADATION",
            "stall_count_in_window": stall_count,
            "total_stall_events": self.stall_events_detected,
            "learned_cooling_efficiency": round(self.learned_cooling_efficiency, 2)
        }
