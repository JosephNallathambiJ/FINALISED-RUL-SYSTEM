import time
from typing import Dict, Any, Callable, Optional

class PowerActuator:
    """
    Executes Power & Voltage Regulation commands (VRM voltage step, relay cut-off).
    Supports custom physical hardware power callbacks.
    """

    def __init__(self):
        self.custom_power_callback: Optional[Callable[[str, float], None]] = None

    def register_power_actuator(self, callback_fn: Callable[[str, float], None]):
        """
        User hook to attach a physical power relay / VRM controller!
        Example:
            actuator.register_power_actuator(lambda action, delta: my_vrm.set_voltage_offset(delta))
        """
        self.custom_power_callback = callback_fn

    def apply_power_action(self, action: str, vcore_delta: float) -> Dict[str, Any]:
        applied_method = "simulated"

        if self.custom_power_callback is not None:
            try:
                self.custom_power_callback(action, vcore_delta)
                applied_method = "custom_hardware_power_callback"
            except Exception:
                applied_method = "custom_error_fallback"

        return {
            "timestamp": time.time(),
            "action": action,
            "vcore_delta": vcore_delta,
            "applied_method": applied_method
        }
