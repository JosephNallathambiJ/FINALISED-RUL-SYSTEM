import time
from typing import Dict, Any, Callable, Optional

class CpuActuator:
    def __init__(self):
        self.custom_freq_actuator: Optional[Callable[[float, str], None]] = None

    def register_freq_actuator(self, callback_fn: Callable[[float, str], None]):
        self.custom_freq_actuator = callback_fn

    def apply_governor(self, target_freq_mhz: float, governor_mode: str) -> Dict[str, Any]:
        method = "simulated_cpufreq"
        if self.custom_freq_actuator is not None:
            try:
                self.custom_freq_actuator(target_freq_mhz, governor_mode)
                method = "custom_hardware_actuator"
            except Exception:
                method = "custom_error_fallback"

        return {
            "timestamp": time.time(),
            "target_freq_mhz": target_freq_mhz,
            "governor_mode": governor_mode,
            "applied_method": method
        }
