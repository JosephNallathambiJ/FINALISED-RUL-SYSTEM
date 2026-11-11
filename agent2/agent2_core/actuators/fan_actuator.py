import time
import glob
from typing import Dict, Any, Callable, Optional

class FanActuator:
    """
    Executes PWM (Pulse Width Modulation) Fan Control commands (0-100%).
    Supports:
    1. SysFS hwmon pwm output (`/sys/class/hwmon/hwmon*/pwm*`)
    2. Custom Hardware PWM Actuator Callback (e.g. Raspberry Pi GPIO PWM pin, Arduino PWM)
    3. Motor Simulation Control
    """

    def __init__(self):
        self.current_pwm_pct: float = 0.0
        self.custom_pwm_callback: Optional[Callable[[float], None]] = None

    def register_pwm_actuator(self, callback_fn: Callable[[float], None]):
        """
        User hook to attach a physical PWM hardware actuator!
        Example:
            actuator.register_pwm_actuator(lambda pwm: my_gpio_pwm.set_duty_cycle(pwm))
        """
        self.custom_pwm_callback = callback_fn

    def set_pwm(self, pwm_pct: float) -> Dict[str, Any]:
        self.current_pwm_pct = max(0.0, min(100.0, round(pwm_pct, 1)))
        applied_method = "simulated"

        if self.custom_pwm_callback is not None:
            try:
                self.custom_pwm_callback(self.current_pwm_pct)
                applied_method = "custom_hardware_pwm_callback"
            except Exception as e:
                applied_method = f"custom_pwm_error_fallback"
        else:
            # Try Linux SysFS hwmon pwm
            try:
                pwm_files = glob.glob("/sys/class/hwmon/hwmon*/pwm1")
                if pwm_files:
                    raw_val = int((self.current_pwm_pct / 100.0) * 255)
                    with open(pwm_files[0], "w") as f:
                        f.write(str(raw_val))
                    applied_method = "sysfs_hwmon_pwm"
            except Exception:
                applied_method = "simulated"

        return {
            "timestamp": time.time(),
            "applied_pwm_pct": self.current_pwm_pct,
            "applied_method": applied_method
        }
