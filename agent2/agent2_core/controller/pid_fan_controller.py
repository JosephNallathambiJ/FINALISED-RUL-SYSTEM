import time
from typing import Dict, Any, Tuple
from agent2_core.config import CoolingProfileConfig

class PidFanController:
    """
    Precision Closed-Loop PID (Proportional-Integral-Derivative) Fan Controller.
    Computes smooth PWM % output to maintain target CPU thermal setpoints.
    """

    def __init__(self, profile: CoolingProfileConfig):
        self.profile = profile
        self.kp = profile.kp
        self.ki = profile.ki
        self.kd = profile.kd
        
        self.integral: float = 0.0
        self.last_error: float = 0.0
        self.last_time: float = time.time()

    def update_profile(self, profile: CoolingProfileConfig):
        self.profile = profile
        self.kp = profile.kp
        self.ki = profile.ki
        self.kd = profile.kd
        self.integral = 0.0
        self.last_error = 0.0

    def compute_pwm_output(self, current_temp: float, target_temp: float) -> Tuple[float, Dict[str, float]]:
        now = time.time()
        dt = max(0.1, min(now - self.last_time, 3.0))
        self.last_time = now

        # Error e(t) = CurrentTemp - TargetTemp
        error = current_temp - target_temp

        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup clamping
        self.integral += error * dt
        self.integral = max(-100.0, min(100.0, self.integral))
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.last_error) / dt
        d_term = self.kd * derivative
        self.last_error = error

        # Raw PID output PWM (0 - 100%)
        raw_pwm = p_term + i_term + d_term

        # Offset baseline for silent threshold
        if current_temp < self.profile.silent_temp_threshold:
            target_pwm = 0.0
        else:
            base_offset = 20.0 if self.profile.min_rpm > 0 else 0.0
            target_pwm = max(base_offset, min(100.0, base_offset + raw_pwm))

        details = {
            "error": round(error, 2),
            "p_term": round(p_term, 2),
            "i_term": round(i_term, 2),
            "d_term": round(d_term, 2),
            "target_pwm_pct": round(target_pwm, 1)
        }
        return round(target_pwm, 1), details
