import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent3_core.config import Agent3Config, load_config
from agent3_core.sensors.system_telemetry import SystemTelemetrySensor
from agent3_core.sensors.voltage_sensor import VoltageSensor
from agent3_core.model.voltage_model import ElectricalDynamicsModel
from agent3_core.utility.power_utility import PowerUtilityEvaluator
from agent3_core.protection.voltage_protector import VoltageProtector
from agent3_core.learning.voltage_history_store import VoltageHistoryStore
from agent3_core.learning.voltage_learner import VoltageLearner
from agent3_core.actuators.power_actuator import PowerActuator
from agent_ml_runtime import AgentMLRuntime

logger = logging.getLogger("Agent3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SystemVoltageAgent:
    """
    Agent 3 - System Voltage & Electrical Power AI Agent.
    Monitors VCore, 12V, 5V, 3.3V, and Battery Cell voltages, preventing brownouts and overvoltage surges.
    """

    def __init__(self, config: Optional[Agent3Config] = None, sensor_mode: str = "auto"):
        self.config = config or load_config()
        self.profile = self.config.current_profile

        # Subsystems
        self.telemetry_sensor = SystemTelemetrySensor()
        self.voltage_sensor = VoltageSensor(
            mode=sensor_mode,
            vcore_nom=self.profile.vcore_nominal,
            v12_nom=self.profile.v12_nominal,
            battery_nom=self.profile.battery_nominal
        )
        self.voltage_model = ElectricalDynamicsModel(
            vcore_nom=self.profile.vcore_nominal,
            vcore_min=self.profile.vcore_min,
            vcore_max=self.profile.vcore_max,
            battery_crit=self.profile.battery_critical
        )
        self.utility_evaluator = PowerUtilityEvaluator(profile=self.profile)
        self.protector = VoltageProtector()
        self.history_store = VoltageHistoryStore(db_path=self.config.db_path)
        self.learner = VoltageLearner(history_store=self.history_store)
        self.actuator = PowerActuator()
        self.ml_runtime = AgentMLRuntime("agent3")

        self.step_count = 0

    def register_voltage_sensor_callback(self, callback_fn: Callable[[], Dict[str, float]]):
        """Attach physical hardware ADC / voltage sensor reading function."""
        self.voltage_sensor.register_voltage_sensor(callback_fn)
        logger.info("Custom hardware Voltage ADC callback registered.")

    def register_power_actuator_callback(self, callback_fn: Callable[[str, float], None]):
        """Attach physical hardware Power / VRM actuator callback function."""
        self.actuator.register_power_actuator(callback_fn)
        logger.info("Custom hardware Power Actuator callback registered.")

    def set_device_profile(self, profile_key: str):
        if profile_key in self.config.profiles:
            self.config.active_profile = profile_key
            self.profile = self.config.current_profile
            self.voltage_sensor.vcore_nom = self.profile.vcore_nominal
            self.voltage_sensor.v12_nom = self.profile.v12_nominal
            self.voltage_sensor.battery_nom = self.profile.battery_nominal
            self.voltage_sensor.simulated_battery_volts = self.profile.battery_nominal if self.profile.battery_nominal > 0 else 12.0
            
            self.voltage_model.vcore_nom = self.profile.vcore_nominal
            self.voltage_model.vcore_min = self.profile.vcore_min
            self.voltage_model.vcore_max = self.profile.vcore_max
            self.voltage_model.battery_crit = self.profile.battery_critical
            
            self.utility_evaluator = PowerUtilityEvaluator(profile=self.profile)
            logger.info(f"Agent3 voltage profile switched to: {self.profile.name}")

    def step(self) -> Dict[str, Any]:
        self.step_count += 1

        # 1. PERCEIVE
        telemetry = self.telemetry_sensor.read()
        cpu_p = telemetry["cpu_percent"]
        voltage_data = self.voltage_sensor.read(cpu_load=cpu_p)
        rails = voltage_data["rails"]

        # 2. MODEL-BASED BELIEF STATE UPDATE
        belief = self.voltage_model.update_belief_state(rails=rails, cpu_p=cpu_p)

        # 3. UTILITY EVALUATION & POWER ACTION SELECTION
        best_action, u_score, utility_details = self.utility_evaluator.select_best_power_action(belief, self.voltage_model)

        # 4. VOLTAGE PROTECTION EXECUTION
        prot_res = self.protector.execute_protection_action(best_action, self.config.active_profile)
        act_res = self.actuator.apply_power_action(best_action, prot_res["vcore_adjust_delta"])

        # 5. HISTORY LOGGING & ONLINE LEARNING
        record = {
            "timestamp": time.time(),
            "vcore_volts": belief.vcore_volts,
            "v12_volts": belief.v12_volts,
            "v5_volts": belief.v5_volts,
            "v33_volts": belief.v33_volts,
            "battery_volts": belief.battery_volts,
            "power_watts": belief.power_watts,
            "utility_score": u_score,
            "voltage_status": belief.voltage_status,
            "action_taken": best_action,
            "device_profile": self.config.active_profile
        }
        self.history_store.record_entry(record)

        learning_info = {}
        if self.step_count % 5 == 0:
            learning_info = self.learner.update_learning_model()

        ml_prediction = self.ml_runtime.predict({
            "step": self.step_count,
            "telemetry": {"cpu_percent": cpu_p, "ram_percent": telemetry.get("ram_percent", 0.0), "battery_percent": telemetry.get("battery_percent", 100.0), "process_count": telemetry.get("process_count", 0), "hw_temperature_c": rails.get("vcore_volts", 0.0) * 10},
            "temperature": {"temperature_c": rails.get("vcore_volts", 0.0) * 10},
            "belief_state": {"temperature_c": rails.get("vcore_volts", 0.0) * 10, "thermal_status": belief.voltage_status},
        })

        return {
            "step": self.step_count,
            "profile": self.profile.name,
            "telemetry": telemetry,
            "rails": rails,
            "belief": {
                "vcore_volts": belief.vcore_volts,
                "v12_volts": belief.v12_volts,
                "v5_volts": belief.v5_volts,
                "v33_volts": belief.v33_volts,
                "battery_volts": belief.battery_volts,
                "power_watts": belief.power_watts,
                "voltage_status": belief.voltage_status,
                "tolerance_deviation_pct": belief.tolerance_deviation_pct
            },
            "utility": utility_details,
            "protection": prot_res,
            "actuator": act_res,
            "learning": learning_info,
            "ml_prediction": ml_prediction,
        }
