import psutil
import time
import random
from typing import Dict, Any, Callable, Optional
from .base_sensor import BaseSensor

class CpuFreqSensor(BaseSensor):
    """
    Monitors CPU Core Frequencies (MHz), DVFS states, and dynamic load scaling.
    """

    def __init__(self, mode: str = "auto", min_freq_mhz: int = 800, max_freq_mhz: int = 3800):
        self.mode = mode.lower()
        self.min_freq = min_freq_mhz
        self.max_freq = max_freq_mhz
        self.simulated_freq: float = (min_freq_mhz + max_freq_mhz) / 2.0
        self.custom_freq_callback: Optional[Callable[[], float]] = None

    def register_freq_sensor(self, callback_fn: Callable[[], float]):
        self.custom_freq_callback = callback_fn
        self.mode = "custom"

    def read_hardware_freq(self) -> Optional[float]:
        try:
            freq = psutil.cpu_freq()
            if freq and freq.current > 0:
                return float(freq.current)
        except Exception:
            pass
        return None

    def simulate_freq_scaling(self, target_governor_freq: float) -> float:
        # Smooth transition towards target governor frequency
        step = (target_governor_freq - self.simulated_freq) * 0.4
        self.simulated_freq += step
        jitter = random.uniform(-10.0, 10.0)
        return round(max(self.min_freq, min(self.max_freq, self.simulated_freq + jitter)), 1)

    def read(self, target_governor_freq: float = 2400.0) -> Dict[str, Any]:
        freq_val: Optional[float] = None
        source_name = "simulated_dvfs"

        if self.mode == "custom" and self.custom_freq_callback is not None:
            try:
                freq_val = self.custom_freq_callback()
                source_name = "custom_hardware_freq_sensor"
            except Exception:
                freq_val = self.simulate_freq_scaling(target_governor_freq)
                source_name = "custom_error_fallback"

        elif self.mode in ("auto", "hardware"):
            hw_freq = self.read_hardware_freq()
            if hw_freq is not None:
                freq_val = hw_freq
                source_name = "os_cpufreq"
            else:
                freq_val = self.simulate_freq_scaling(target_governor_freq)
                source_name = "simulated_dvfs"
        else:
            freq_val = self.simulate_freq_scaling(target_governor_freq)
            source_name = "simulated_dvfs"

        return {
            "timestamp": time.time(),
            "freq_mhz": freq_val,
            "min_freq_mhz": self.min_freq,
            "max_freq_mhz": self.max_freq,
            "sensor_source": source_name
        }
