import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent4_core.config import Agent4Config, load_config
from agent4_core.sensors.system_telemetry import SystemTelemetrySensor
from agent4_core.sensors.cpu_freq_sensor import CpuFreqSensor
from agent4_core.model.cpu_model import CpuWorkloadModel
from agent4_core.utility.cpu_utility import CpuUtilityEvaluator
from agent4_core.controller.dvfs_governor import DvfsGovernor
from agent4_core.learning.cpu_history_store import CpuHistoryStore
from agent4_core.learning.workload_learner import WorkloadLearner
from agent4_core.actuators.cpu_actuator import CpuActuator
from agent_ml_runtime import AgentMLRuntime

logger = logging.getLogger("Agent4")

class CpuPerformanceAgent:
    def __init__(self, config: Optional[Agent4Config] = None):
        self.config = config or load_config()
        self.profile = self.config.current_profile

        self.telemetry = SystemTelemetrySensor()
        self.freq_sensor = CpuFreqSensor(
            min_freq_mhz=self.profile.min_freq_mhz,
            max_freq_mhz=self.profile.max_freq_mhz
        )
        self.model = CpuWorkloadModel(
            min_freq=self.profile.min_freq_mhz,
            max_freq=self.profile.max_freq_mhz,
            target_load=self.profile.target_cpu_load
        )
        self.utility_evaluator = CpuUtilityEvaluator(profile=self.profile)
        self.governor = DvfsGovernor(profile=self.profile)
        self.history_store = CpuHistoryStore(db_path=self.config.db_path)
        self.learner = WorkloadLearner(history_store=self.history_store)
        self.actuator = CpuActuator()
        self.ml_runtime = AgentMLRuntime("agent4")

        self.step_count = 0
        self.last_target_freq = 2400.0

    def set_device_profile(self, profile_key: str):
        if profile_key in self.config.profiles:
            self.config.active_profile = profile_key
            self.profile = self.config.current_profile
            self.freq_sensor.min_freq = self.profile.min_freq_mhz
            self.freq_sensor.max_freq = self.profile.max_freq_mhz
            self.model.min_freq = self.profile.min_freq_mhz
            self.model.max_freq = self.profile.max_freq_mhz
            self.model.target_load = self.profile.target_cpu_load
            self.utility_evaluator = CpuUtilityEvaluator(profile=self.profile)
            self.governor = DvfsGovernor(profile=self.profile)

    def step(self) -> Dict[str, Any]:
        self.step_count += 1

        telem = self.telemetry.read()
        cpu_p = telem["cpu_percent"]
        freq_data = self.freq_sensor.read(target_governor_freq=self.last_target_freq)
        current_freq = freq_data["freq_mhz"]

        belief = self.model.update_belief_state(current_freq, cpu_p, self.profile.governor_mode)
        best_freq, u_score, utility_details = self.utility_evaluator.select_best_frequency(belief)

        gov_res = self.governor.compute_governor_action(current_freq, cpu_p)
        self.last_target_freq = gov_res["target_freq_mhz"]

        act_res = self.actuator.apply_governor(gov_res["target_freq_mhz"], gov_res["governor_mode"])

        record = {
            "timestamp": time.time(),
            "freq_mhz": current_freq,
            "cpu_percent": cpu_p,
            "governor_mode": gov_res["governor_mode"],
            "utility_score": u_score,
            "device_profile": self.config.active_profile
        }
        self.history_store.record_entry(record)

        learning_info = {}
        if self.step_count % 5 == 0:
            learning_info = self.learner.update_learning_model()

        ml_prediction = self.ml_runtime.predict({
            "step": self.step_count,
            "telemetry": {"cpu_percent": cpu_p, "ram_percent": telem.get("ram_percent", 0.0), "battery_percent": telem.get("battery_percent", 100.0), "process_count": telem.get("process_count", 0), "hw_temperature_c": current_freq / 10.0},
            "temperature": {"temperature_c": current_freq / 10.0},
            "belief_state": {"temperature_c": current_freq / 10.0, "thermal_status": belief.perf_status},
        })

        return {
            "step": self.step_count,
            "profile": self.profile.name,
            "telemetry": telem,
            "freq_data": freq_data,
            "belief": {
                "freq_mhz": belief.current_freq_mhz,
                "cpu_percent": belief.cpu_percent,
                "perf_status": belief.perf_status
            },
            "utility": utility_details,
            "governor": gov_res,
            "actuator": act_res,
            "learning": learning_info,
            "ml_prediction": ml_prediction,
        }
