import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent2_core.config import Agent2Config, load_config
from agent2_core.sensors.system_telemetry import SystemTelemetrySensor
from agent2_core.sensors.fan_rpm_sensor import FanRpmSensor
from agent2_core.model.cooling_model import CoolingDynamicsModel
from agent2_core.utility.fan_utility import FanUtilityEvaluator
from agent2_core.controller.pid_fan_controller import PidFanController
from agent2_core.learning.rpm_history_store import RpmHistoryStore
from agent2_core.learning.fan_learner import FanLearner
from agent2_core.actuators.fan_actuator import FanActuator
from agent_ml_runtime import AgentMLRuntime

logger = logging.getLogger("Agent2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class FanCoolingAgent:
    """
    Agent 2 - Fan RPM & Active Cooling AI Agent.
    Monitors CPU/IoT thermal state and controls CPU Fan RPM, PWM Duty Cycles, and Acoustics.
    """

    def __init__(self, config: Optional[Agent2Config] = None, sensor_mode: str = "auto"):
        self.config = config or load_config()
        self.profile = self.config.current_profile

        # Subsystems
        self.telemetry_sensor = SystemTelemetrySensor()
        self.fan_sensor = FanRpmSensor(mode=sensor_mode, max_rpm=self.profile.max_rpm)
        self.cooling_model = CoolingDynamicsModel(
            max_rpm=self.profile.max_rpm,
            target_temp=self.profile.target_temp,
            max_noise_dba=self.profile.max_noise_dba
        )
        self.utility_evaluator = FanUtilityEvaluator(profile=self.profile)
        self.pid_controller = PidFanController(profile=self.profile)
        self.history_store = RpmHistoryStore(db_path=self.config.db_path)
        self.learner = FanLearner(history_store=self.history_store)
        self.actuator = FanActuator()
        self.ml_runtime = AgentMLRuntime("agent2")

        self.step_count = 0
        self.last_target_pwm = 0.0

    def register_rpm_sensor_callback(self, callback_fn: Callable[[], float]):
        """Attach physical hardware fan tachometer RPM reading function."""
        self.fan_sensor.register_rpm_sensor(callback_fn)
        logger.info("Custom hardware Fan Tachometer callback registered.")

    def register_pwm_actuator_callback(self, callback_fn: Callable[[float], None]):
        """Attach physical hardware PWM fan control output function."""
        self.actuator.register_pwm_actuator(callback_fn)
        logger.info("Custom hardware PWM Fan Actuator callback registered.")

    def set_device_profile(self, profile_key: str):
        if profile_key in self.config.profiles:
            self.config.active_profile = profile_key
            self.profile = self.config.current_profile
            self.fan_sensor.max_rpm = self.profile.max_rpm
            self.cooling_model.max_rpm = self.profile.max_rpm
            self.cooling_model.target_temp = self.profile.target_temp
            self.cooling_model.max_noise_dba = self.profile.max_noise_dba
            self.utility_evaluator = FanUtilityEvaluator(profile=self.profile)
            self.pid_controller.update_profile(self.profile)
            logger.info(f"Agent2 cooling profile switched to: {self.profile.name}")

    def step(self) -> Dict[str, Any]:
        self.step_count += 1

        # 1. PERCEIVE
        telemetry = self.telemetry_sensor.read()
        cpu_p = telemetry["cpu_percent"]
        
        # Read temperature (use hardware or fallback simulator in model)
        if telemetry["hw_temperature_c"] is not None:
            current_temp = telemetry["hw_temperature_c"]
        else:
            # Simulate thermal dynamic response to fan cooling airflow
            cfm = self.cooling_model.compute_airflow_cfm(self.fan_sensor.simulated_rpm)
            heat = (cpu_p / 100.0) * 4.5
            cool = (self.cooling_model.simulated_temp - 25.0) * 0.04 + (cfm * 0.15)
            self.cooling_model.simulated_temp += (heat - cool) * 0.5
            current_temp = round(max(25.0, self.cooling_model.simulated_temp), 2)

        # Read Fan RPM tachometer
        fan_data = self.fan_sensor.read(target_pwm_percent=self.last_target_pwm)
        current_rpm = fan_data["rpm"]

        # 2. MODEL-BASED BELIEF STATE UPDATE
        belief = self.cooling_model.update_belief_state(
            current_rpm=current_rpm,
            pwm_pct=self.last_target_pwm,
            current_temp=current_temp,
            cpu_p=cpu_p
        )

        # 3. UTILITY EVALUATION & TARGET RPM SELECTION
        optimal_rpm, u_score, utility_details = self.utility_evaluator.select_optimal_rpm(belief, self.cooling_model)

        # 4. CLOSED-LOOP PID CONTROLLER PWM COMPUTATION
        pid_pwm, pid_details = self.pid_controller.compute_pwm_output(current_temp, self.profile.target_temp)
        self.last_target_pwm = pid_pwm

        # 5. ACTUATE PWM TO FAN MOTOR
        actuation_res = self.actuator.set_pwm(pid_pwm)

        # 6. HISTORY RECORDING & ONLINE LEARNING
        record = {
            "timestamp": time.time(),
            "current_rpm": current_rpm,
            "target_pwm_pct": pid_pwm,
            "temperature_c": current_temp,
            "cpu_percent": cpu_p,
            "airflow_cfm": belief.airflow_cfm,
            "noise_dba": belief.acoustic_noise_dba,
            "utility_score": u_score,
            "fan_health": belief.fan_health_status,
            "device_profile": self.config.active_profile
        }
        self.history_store.record_entry(record)

        learning_info = {}
        if self.step_count % 5 == 0:
            learning_info = self.learner.update_learning_model()

        ml_prediction = self.ml_runtime.predict({
            "step": self.step_count,
            "telemetry": {"cpu_percent": cpu_p, "ram_percent": telemetry.get("ram_percent", 0.0), "battery_percent": telemetry.get("battery_percent", 100.0), "process_count": telemetry.get("process_count", 0), "hw_temperature_c": current_temp},
            "temperature": {"temperature_c": current_temp},
            "belief_state": {"temperature_c": current_temp, "thermal_status": belief.thermal_status},
        })

        return {
            "step": self.step_count,
            "profile": self.profile.name,
            "current_rpm": current_rpm,
            "target_pwm_pct": pid_pwm,
            "temperature_c": current_temp,
            "cpu_percent": cpu_p,
            "airflow_cfm": belief.airflow_cfm,
            "noise_dba": belief.acoustic_noise_dba,
            "thermal_status": belief.thermal_status,
            "fan_health": belief.fan_health_status,
            "utility": utility_details,
            "pid": pid_details,
            "actuator": actuation_res,
            "learning": learning_info,
            "ml_prediction": ml_prediction,
        }
