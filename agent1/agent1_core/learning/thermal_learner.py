import math
from typing import Dict, Any, List
from agent1_core.learning.history_store import TemperatureHistoryStore

class ThermalLearner:
    """
    Learning Agent Subsystem:
    Analyzes past history of all temperature readings to build dynamic baselines,
    learn thermal time constants, perform anomaly detection, and tune future expectations.
    """

    def __init__(self, history_store: TemperatureHistoryStore):
        self.history_store = history_store
        self.learned_baseline_temp: float = 30.0
        self.learned_std_dev: float = 5.0
        self.thermal_anomalies_detected: int = 0
        self.heating_rate_coef: float = 0.35 # Learned °C per CPU%

    def update_learning_model(self) -> Dict[str, Any]:
        """
        Runs online learning pass over stored temperature history.
        """
        recent_history = self.history_store.get_recent_history(limit=200)
        if not recent_history:
            return {
                "learned_baseline_c": self.learned_baseline_temp,
                "learned_std_dev": self.learned_std_dev,
                "anomalies_count": self.thermal_anomalies_detected,
                "status": "INSUFFICIENT_DATA"
            }

        temps = [r["temperature_c"] for r in recent_history]
        n = len(temps)
        avg_temp = sum(temps) / n

        variance = sum((x - avg_temp) ** 2 for x in temps) / max(1, n - 1)
        std_dev = math.sqrt(variance)

        self.learned_baseline_temp = round(avg_temp, 2)
        self.learned_std_dev = round(std_dev, 2)

        # Estimate heating rate correlation between CPU and temperature velocity
        cpu_temp_pairs = [(r["cpu_percent"], r["thermal_velocity"]) for r in recent_history if "thermal_velocity" in r]
        if len(cpu_temp_pairs) > 10:
            positive_slopes = [v / max(1.0, c) for c, v in cpu_temp_pairs if c > 20 and v > 0]
            if positive_slopes:
                self.heating_rate_coef = round(sum(positive_slopes) / len(positive_slopes), 4)

        return {
            "learned_baseline_c": self.learned_baseline_temp,
            "learned_std_dev": self.learned_std_dev,
            "heating_rate_coef": self.heating_rate_coef,
            "total_samples_learned": n,
            "status": "MODEL_UPDATED"
        }

    def detect_anomaly(self, current_temp: float, current_velocity: float) -> Dict[str, Any]:
        """
        Anomaly Detection using 2.5-sigma bounds and rapid velocity check.
        """
        upper_bound = self.learned_baseline_temp + (2.5 * max(2.0, self.learned_std_dev))
        is_temp_anomaly = current_temp > upper_bound
        is_velocity_spike = current_velocity > 1.5  # Rising faster than 1.5°C/sec

        is_anomaly = is_temp_anomaly or is_velocity_spike
        if is_anomaly:
            self.thermal_anomalies_detected += 1

        return {
            "is_anomaly": is_anomaly,
            "anomaly_type": "TEMPERATURE_SPIKE" if is_temp_anomaly else ("VELOCITY_SPIKE" if is_velocity_spike else "NORMAL"),
            "upper_bound_c": round(upper_bound, 2),
            "current_temp_c": current_temp,
            "total_anomalies_logged": self.thermal_anomalies_detected
        }
