import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent1_core.config import AgentConfig, load_config
from agent1_core.sensors.system_telemetry import SystemTelemetrySensor
from agent1_core.sensors.temperature_sensor import TemperatureSensor
from agent1_core.model.world_model import WorldModel
from agent1_core.utility.utility_evaluator import UtilityEvaluator
from agent1_core.learning.history_store import TemperatureHistoryStore
from agent1_core.learning.thermal_learner import ThermalLearner
from agent1_core.actuators.system_actuator import SystemActuator
from agent_ml_runtime import AgentMLRuntime

logger = logging.getLogger("Agent1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ModelUtilityLearningAgent:
    """
    Model-Based, Utility-Based, Learning AI Agent ('agent1').
    Monitors CPU/IoT telemetry across Laptops, Desktops, Drones, and IoT systems.
    """

    def __init__(self, config: Optional[AgentConfig] = None, sensor_mode: str = "auto"):
        self.config = config or load_config()
        self.profile = self.config.current_profile

        # Perception Subsystem
        self.telemetry_sensor = SystemTelemetrySensor()
        self.temp_sensor = TemperatureSensor(mode=sensor_mode)

        # Model-Based Subsystem
        self.world_model = WorldModel(
            warning_temp=self.profile.thermal_warning_temp,
            critical_temp=self.profile.thermal_critical_temp
        )

        # Utility-Based Subsystem
        self.utility_evaluator = UtilityEvaluator(profile=self.profile)

        # Learning & History Subsystem
        self.history_store = TemperatureHistoryStore(db_path=self.config.db_path)
        self.learner = ThermalLearner(history_store=self.history_store)

        # Actuation Subsystem
        self.actuator = SystemActuator()

        # Runtime inference model
        self.ml_runtime = AgentMLRuntime("agent1")

        self.step_count = 0
        self.last_action_cooling = 0.0

    def register_hardware_temperature_sensor(self, callback_fn: Callable[[], float]):
        """
        User extension hook for attaching physical temperature sensors (I2C, SPI, Serial, etc.).
        """
        self.temp_sensor.register_custom_sensor(callback_fn)
        logger.info("Custom physical hardware temperature sensor callback registered successfully.")

    def set_device_profile(self, profile_key: str):
        """Switch active profile (e.g. 'drone', 'laptop', 'computer', 'iot')."""
        if profile_key in self.config.profiles:
            self.config.active_profile = profile_key
            self.profile = self.config.current_profile
            self.utility_evaluator = UtilityEvaluator(profile=self.profile)
            self.world_model.warning_temp = self.profile.thermal_warning_temp
            self.world_model.critical_temp = self.profile.thermal_critical_temp
            logger.info(f"Agent profile switched to: {self.profile.name}")

    def step(self) -> Dict[str, Any]:
        """
        Performs 1 complete Agent Execution Loop:
        1. PERCEIVE: Read system telemetry & temperature sensor
        2. MODEL: Update belief state & compute physics transition model
        3. UTILITY: Evaluate multi-attribute utility U(S, A) for actions and select A*
        4. ACT: Execute action A*
        5. LEARN: Store to temperature history & update online thermal learner
        """
        self.step_count += 1
        
        # 1. PERCEIVE
        telemetry = self.telemetry_sensor.read()
        cpu_load = telemetry["cpu_percent"]
        temp_data = self.temp_sensor.read(
            current_cpu_load=cpu_load,
            action_cooling=self.last_action_cooling
        )
        temp_c = temp_data["temperature_celsius"]

        # 2. MODEL-BASED BELIEF STATE UPDATE
        belief_state = self.world_model.update_state(
            telemetry=telemetry,
            temp_data=temp_data,
            profile_name=self.profile.name
        )

        # 3. UTILITY-BASED DECISION
        best_action, best_utility, utility_details = self.utility_evaluator.select_best_action(
            state=belief_state,
            world_model=self.world_model
        )
        belief_state.current_action = best_action

        # 4. ACTUATION
        actuation_result = self.actuator.execute_action(best_action, self.config.active_profile)
        self.last_action_cooling = actuation_result["action_cooling_effect"]

        # 5. LEARNING & TEMPERATURE HISTORY RECORDING
        history_record = {
            "timestamp": time.time(),
            "temperature_c": temp_c,
            "cpu_percent": cpu_load,
            "ram_percent": telemetry["ram_percent"],
            "battery_percent": telemetry["battery_percent"],
            "thermal_velocity": belief_state.temp_velocity_c_per_sec,
            "action_taken": best_action,
            "utility_score": best_utility,
            "sensor_source": temp_data["sensor_source"],
            "thermal_status": belief_state.thermal_status,
            "device_profile": self.config.active_profile
        }
        self.history_store.record_entry(history_record)

        # Run online learning update every 5 steps
        learning_info = {}
        if self.step_count % 5 == 0:
            learning_info = self.learner.update_learning_model()

        # Check anomaly
        anomaly_info = self.learner.detect_anomaly(temp_c, belief_state.temp_velocity_c_per_sec)

        ml_prediction = self.ml_runtime.predict({
            "step": self.step_count,
            "telemetry": telemetry,
            "temperature": temp_data,
            "belief_state": {
                "temperature_c": belief_state.temperature_c,
                "velocity_c_per_sec": belief_state.temp_velocity_c_per_sec,
                "acceleration": belief_state.temp_acceleration,
                "thermal_status": belief_state.thermal_status,
            },
        })

        return {
            "step": self.step_count,
            "profile": self.profile.name,
            "telemetry": telemetry,
            "temperature": temp_data,
            "belief_state": {
                "temperature_c": belief_state.temperature_c,
                "velocity_c_per_sec": belief_state.temp_velocity_c_per_sec,
                "acceleration": belief_state.temp_acceleration,
                "thermal_status": belief_state.thermal_status
            },
            "utility": utility_details,
            "action": actuation_result,
            "anomaly": anomaly_info,
            "learning": learning_info,
            "ml_prediction": ml_prediction,
        }
