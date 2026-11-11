import time
from typing import Dict, Any
from agent4_core.config import CpuProfileConfig

class DvfsGovernor:
    def __init__(self, profile: CpuProfileConfig):
        self.profile = profile

    def compute_governor_action(self, current_freq: float, cpu_load: float) -> Dict[str, Any]:
        target_freq = current_freq
        mode = self.profile.governor_mode

        if cpu_load > 80.0:
            target_freq = float(self.profile.max_freq_mhz)
            mode = "PERFORMANCE"
        elif cpu_load < 25.0:
            target_freq = float(self.profile.min_freq_mhz)
            mode = "POWERSAVE"
        else:
            ratio = (cpu_load / 100.0)
            target_freq = self.profile.min_freq_mhz + ratio * (self.profile.max_freq_mhz - self.profile.min_freq_mhz)
            mode = "SCHEDUTIL"

        return {
            "timestamp": time.time(),
            "target_freq_mhz": round(target_freq, 1),
            "governor_mode": mode
        }
