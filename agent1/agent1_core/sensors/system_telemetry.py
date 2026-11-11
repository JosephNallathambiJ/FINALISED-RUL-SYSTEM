import psutil
import time
from typing import Dict, Any
from .base_sensor import BaseSensor

class SystemTelemetrySensor(BaseSensor):
    """Monitors CPU load, memory usage, battery, frequency, and process statistics."""

    def __init__(self):
        # Warmup psutil CPU percent calculation
        psutil.cpu_percent(interval=None)

    def read(self) -> Dict[str, Any]:
        cpu_overall = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        memory_info = psutil.virtual_memory()
        
        # CPU Frequency if available
        cpu_freq_current = 0.0
        cpu_freq_max = 0.0
        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_freq_current = freq.current
                cpu_freq_max = freq.max
        except Exception:
            pass

        # Battery status if available (laptops, mobile edge devices, drones)
        battery_pct = 100.0
        is_charging = True
        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_pct = battery.percent
                is_charging = battery.power_plugged if battery.power_plugged is not None else True
        except Exception:
            pass

        process_count = len(psutil.pids())

        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_overall,
            "cpu_per_core": cpu_per_core,
            "cpu_freq_mhz": cpu_freq_current,
            "cpu_freq_max_mhz": cpu_freq_max,
            "ram_percent": memory_info.percent,
            "ram_used_gb": round(memory_info.used / (1024 ** 3), 2),
            "ram_total_gb": round(memory_info.total / (1024 ** 3), 2),
            "battery_percent": battery_pct,
            "is_charging": is_charging,
            "process_count": process_count
        }
