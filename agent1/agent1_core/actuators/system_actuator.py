import time
from typing import Dict, Any

class SystemActuator:
    """
    Executes physical or simulated system adjustments based on agent action decisions.
    """

    def __init__(self):
        self.current_fan_speed_pct: float = 20.0
        self.current_cpu_throttle_pct: float = 0.0
        self.last_action_time: float = time.time()
        self.action_history = []

    def execute_action(self, action: str, profile_name: str) -> Dict[str, Any]:
        """Executes selected control action and returns status payload."""
        now = time.time()
        self.last_action_time = now

        msg = ""
        cooling_delta = 0.0

        if action == "PASSIVE":
            self.current_fan_speed_pct = 20.0
            self.current_cpu_throttle_pct = 0.0
            cooling_delta = 0.0
            msg = "System operating normally. Passive thermal equilibrium."

        elif action == "COOL_MODERATE":
            self.current_fan_speed_pct = 60.0
            self.current_cpu_throttle_pct = 0.0
            cooling_delta = 0.5
            msg = f"[{profile_name.upper()}] Cooling fan speed boosted to 60%."

        elif action == "THROTTLE_LIGHT":
            self.current_fan_speed_pct = 80.0
            self.current_cpu_throttle_pct = 20.0
            cooling_delta = 1.0
            msg = f"[{profile_name.upper()}] Light CPU frequency scaling applied (-20% load cap). Fan at 80%."

        elif action == "THROTTLE_HEAVY":
            self.current_fan_speed_pct = 100.0
            self.current_cpu_throttle_pct = 50.0
            cooling_delta = 2.0
            msg = f"[{profile_name.upper()}] Heavy thermal throttling active (-50% load cap). Fan at 100%."

        elif action == "REDUCE_PAYLOAD":
            self.current_fan_speed_pct = 90.0
            self.current_cpu_throttle_pct = 35.0
            cooling_delta = 1.5
            msg = f"[DRONE PAYLOAD] Shedding auxiliary payload tasks to prevent flight controller thermal shutdown."

        elif action == "EMERGENCY_COOLING":
            self.current_fan_speed_pct = 100.0
            self.current_cpu_throttle_pct = 75.0
            cooling_delta = 3.5
            msg = f"[CRITICAL WARNING] Emergency cooling engaged! CPU throttled by 75%."

        elif action == "EMERGENCY_LANDING_ALERT":
            self.current_fan_speed_pct = 100.0
            self.current_cpu_throttle_pct = 80.0
            cooling_delta = 4.0
            msg = f"[DRONE EMERGENCY] CRITICAL OVERHEAT! Initiating Emergency Landing Protocol broadcast!"

        elif action == "SLEEP_DUTY_CYCLE":
            self.current_fan_speed_pct = 0.0
            self.current_cpu_throttle_pct = 85.0
            cooling_delta = 2.5
            msg = f"[IoT EDGE] Entering low-power duty cycle sleep mode to dissipate heat."

        status = {
            "timestamp": now,
            "executed_action": action,
            "fan_speed_percent": self.current_fan_speed_pct,
            "cpu_throttle_percent": self.current_cpu_throttle_pct,
            "action_cooling_effect": cooling_delta,
            "message": msg
        }
        self.action_history.append(status)
        if len(self.action_history) > 50:
            self.action_history.pop(0)
        return status
