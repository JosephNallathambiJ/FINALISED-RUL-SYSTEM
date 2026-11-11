import psutil
import time
from typing import Dict, Any
from .base_sensor import BaseSensor

class SystemTelemetrySensor(BaseSensor):
    def __init__(self):
        psutil.cpu_percent(interval=None)

    def read(self) -> Dict[str, Any]:
        cpu_overall = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        memory_info = psutil.virtual_memory()

        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_overall,
            "cpu_per_core": cpu_per_core,
            "ram_percent": memory_info.percent,
            "process_count": len(psutil.pids())
        }
