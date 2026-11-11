import time
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CpuStateBelief:
    timestamp: float
    current_freq_mhz: float
    cpu_percent: float
    target_load: float
    governor_mode: str
    perf_status: str  # "OPTIMAL", "THROTTLED", "HIGH_PERF", "OVERLOAD"

class CpuWorkloadModel:
    def __init__(self, min_freq: int = 800, max_freq: int = 3800, target_load: float = 65.0):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.target_load = target_load

    def update_belief_state(self, current_freq: float, cpu_p: float, gov_mode: str) -> CpuStateBelief:
        if cpu_p > 88.0:
            status = "OVERLOAD"
        elif current_freq < (self.min_freq + 200):
            status = "THROTTLED"
        elif cpu_p > 60.0:
            status = "HIGH_PERF"
        else:
            status = "OPTIMAL"

        return CpuStateBelief(
            timestamp=time.time(),
            current_freq_mhz=current_freq,
            cpu_percent=cpu_p,
            target_load=self.target_load,
            governor_mode=gov_mode,
            perf_status=status
        )
