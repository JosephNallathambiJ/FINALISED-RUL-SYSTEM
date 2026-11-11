import psutil
import time
from typing import Dict, Any
from .base_sensor import BaseSensor

class SystemTelemetrySensor(BaseSensor):
    """Monitors CPU load, temperature, battery, and process telemetry."""

    def __init__(self):
        psutil.cpu_percent(interval=None)

    def read(self) -> Dict[str, Any]:
        cpu_overall = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        
        # Read temperature if hardware sensor available, else None
        hw_temp = None
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                val_list = [e.current for n, entries in temps.items() for e in entries if e.current]
                if val_list:
                    hw_temp = max(val_list)
        except Exception:
            pass

        battery_pct = 100.0
        is_charging = True
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_pct = battery.percent
                is_charging = battery.power_plugged if battery.power_plugged is not None else True
        except Exception:
            pass

        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_overall,
            "hw_temperature_c": hw_temp,
            "ram_percent": memory_info.percent,
            "battery_percent": battery_pct,
            "is_charging": is_charging
        }
