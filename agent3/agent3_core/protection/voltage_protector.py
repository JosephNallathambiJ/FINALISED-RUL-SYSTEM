import time
from typing import Dict, Any

class VoltageProtector:
    """
    Translates selected agent actions into physical or simulated voltage protection commands.
    """

    def __init__(self):
        self.last_action_time = time.time()

    def execute_protection_action(self, action: str, profile_name: str) -> Dict[str, Any]:
        now = time.time()
        self.last_action_time = now

        msg = ""
        vcore_adjust_delta = 0.0

        if action == "STABLE_PASSIVE":
            vcore_adjust_delta = 0.0
            msg = f"[{profile_name.upper()}] Multi-rail voltages operating smoothly within ±5% tolerance."

        elif action == "BOOST_VCORE":
            vcore_adjust_delta = +0.03
            msg = f"[{profile_name.upper()}] Boosting VCore VRM supply (+0.03V) to mitigate voltage sag."

        elif action == "CAP_POWER_DRAW":
            vcore_adjust_delta = +0.01
            msg = f"[{profile_name.upper()}] Capping CPU current draw to stabilize power supply rail."

        elif action == "SHED_LOAD_BROWNOUT":
            vcore_adjust_delta = +0.05
            msg = f"[BROWNOUT PROTECTION] Shedding non-critical background jobs to prevent system brownout crash."

        elif action == "TRIM_VOLTAGE_SURGE":
            vcore_adjust_delta = -0.04
            msg = f"[SURGE PROTECTION] Trimming VRM output (-0.04V) to suppress voltage spike."

        elif action == "EMERGENCY_VOLTAGE_TRIP":
            vcore_adjust_delta = 0.0
            msg = f"[CRITICAL POWER EMERGENCY] Extreme Voltage Anomaly! Trip Protection Alert Broadcasted!"

        return {
            "timestamp": now,
            "executed_action": action,
            "vcore_adjust_delta": vcore_adjust_delta,
            "message": msg
        }
