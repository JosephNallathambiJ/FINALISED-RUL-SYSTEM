import math
from typing import Dict, Any
from agent3_core.learning.voltage_history_store import VoltageHistoryStore

class VoltageLearner:
    """
    Learning Agent Subsystem for System Voltages:
    Monitors voltage history to build baseline ripple statistics and detect transients (Sag / Surge).
    """

    def __init__(self, history_store: VoltageHistoryStore):
        self.history_store = history_store
        self.transient_sags_detected: int = 0
        self.transient_surges_detected: int = 0
        self.learned_vcore_ripple: float = 0.015

    def update_learning_model(self) -> Dict[str, Any]:
        history = self.history_store.get_recent_history(limit=100)
        if len(history) < 10:
            return {"status": "COLLECTING_DATA", "ripple_volts": self.learned_vcore_ripple}

        vcores = [r["vcore_volts"] for r in history]
        avg_v = sum(vcores) / len(vcores)
        variance = sum((x - avg_v) ** 2 for x in vcores) / max(1, len(vcores) - 1)
        ripple = math.sqrt(variance)

        self.learned_vcore_ripple = round(ripple, 4)

        # Count transients
        sags = sum(1 for r in history if r["voltage_status"] in ("UNDER_VOLTAGE_SAG", "BROWNOUT_RISK"))
        surges = sum(1 for r in history if r["voltage_status"] == "OVER_VOLTAGE_SURGE")

        self.transient_sags_detected += sags
        self.transient_surges_detected += surges

        return {
            "status": "STABLE" if (sags + surges) == 0 else "TRANSIENTS_DETECTED",
            "learned_ripple_volts": self.learned_vcore_ripple,
            "sags_count": sags,
            "surges_count": surges,
            "total_sags_logged": self.transient_sags_detected,
            "total_surges_logged": self.transient_surges_detected
        }
