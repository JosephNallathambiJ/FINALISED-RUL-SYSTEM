import psutil
import time
import glob
import random
from typing import Dict, Any, Callable, Optional
from .base_sensor import BaseSensor

class FanRpmSensor(BaseSensor):
    """
    Fan Tachometer & RPM Sensing Module:
    1. Reads OS/SysFS hwmon Fan Speed (RPM)
    2. Supports Custom Hardware RPM Tachometer Callbacks (GPIO, Arduino, I2C tach)
    3. Dynamic Fan Motor Physics Simulator (models PWM -> RPM rotational dynamics)
    """

    def __init__(self, mode: str = "auto", max_rpm: int = 5000):
        self.mode = mode.lower()
        self.max_rpm = max_rpm
        self.simulated_rpm: float = 0.0
        self.custom_rpm_callback: Optional[Callable[[], float]] = None
        self._last_time = time.time()

    def register_rpm_sensor(self, callback_fn: Callable[[], float]):
        """
        User hook to attach a physical fan tachometer / RPM sensor!
        Example:
            sensor.register_rpm_sensor(lambda: my_gpio_tachometer.read_rpm())
        """
        self.custom_rpm_callback = callback_fn
        self.mode = "custom"

    def read_hardware_rpm(self) -> Optional[int]:
        """Attempts to read actual Fan RPM from psutil or Linux hwmon sysfs."""
        try:
            fans = psutil.sensors_fans()
            if fans:
                for name, entries in fans.items():
                    for entry in entries:
                        if entry.current and entry.current > 0:
                            return int(entry.current)
        except Exception:
            pass

        # Try reading Linux hwmon fan*_input
        try:
            fan_files = glob.glob("/sys/class/hwmon/hwmon*/fan*_input")
            for fpath in fan_files:
                with open(fpath, "r") as f:
                    val = f.read().strip()
                    if val.isdigit() and int(val) > 0:
                        return int(val)
        except Exception:
            pass

        return None

    def simulate_fan_motor_physics(self, target_pwm_percent: float) -> int:
        """
        Rotational inertia differential equation:
        dRPM/dt = (Target_RPM - Current_RPM) / tau
        """
        now = time.time()
        dt = max(0.05, min(now - self._last_time, 2.0))
        self._last_time = now

        target_rpm = (target_pwm_percent / 100.0) * self.max_rpm
        
        # Rotational time constant tau (sec)
        tau = 0.45
        step = (target_rpm - self.simulated_rpm) * (dt / tau)
        self.simulated_rpm += step
        
        # Add motor jitter when spinning
        if self.simulated_rpm > 100:
            jitter = random.uniform(-15.0, 15.0)
        else:
            jitter = 0.0

        current_rpm = max(0, int(self.simulated_rpm + jitter))
        return current_rpm

    def read(self, target_pwm_percent: float = 0.0) -> Dict[str, Any]:
        rpm_val: Optional[int] = None
        source_name = "simulated_motor_physics"

        if self.mode == "custom" and self.custom_rpm_callback is not None:
            try:
                rpm_val = int(self.custom_rpm_callback())
                source_name = "custom_hardware_tachometer"
            except Exception:
                rpm_val = self.simulate_fan_motor_physics(target_pwm_percent)
                source_name = "custom_error_fallback"

        elif self.mode in ("auto", "hardware"):
            hw_rpm = self.read_hardware_rpm()
            if hw_rpm is not None:
                rpm_val = hw_rpm
                source_name = "os_hwmon_fan_tachometer"
            else:
                rpm_val = self.simulate_fan_motor_physics(target_pwm_percent)
                source_name = "simulated_motor_physics"
        else:
            rpm_val = self.simulate_fan_motor_physics(target_pwm_percent)
            source_name = "simulated_motor_physics"

        return {
            "timestamp": time.time(),
            "rpm": rpm_val,
            "max_rpm": self.max_rpm,
            "sensor_source": source_name,
            "is_custom_attached": (self.custom_rpm_callback is not None)
        }
