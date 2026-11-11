import time
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class SystemBeliefState:
    timestamp: float
    temperature_c: float
    temp_velocity_c_per_sec: float  # dT/dt
    temp_acceleration: float        # d2T/dt2
    cpu_percent: float
    ram_percent: float
    battery_percent: float
    is_charging: bool
    current_action: str
    thermal_status: str             # "NORMAL", "WARM", "CRITICAL", "OVERHEAT"
    device_profile_name: str

class WorldModel:
    """
    Model-Based Agent Subsystem:
    Maintains internal model of system dynamics, thermal inertia, and predicts future states.
    """

    def __init__(self, warning_temp: float = 70.0, critical_temp: float = 85.0):
        self.warning_temp = warning_temp
        self.critical_temp = critical_temp
        
        self.last_temp: float = 25.0
        self.last_velocity: float = 0.0
        self.last_time: float = time.time()
        self.current_state: SystemBeliefState = SystemBeliefState(
            timestamp=time.time(),
            temperature_c=25.0,
            temp_velocity_c_per_sec=0.0,
            temp_acceleration=0.0,
            cpu_percent=0.0,
            ram_percent=0.0,
            battery_percent=100.0,
            is_charging=True,
            current_action="PASSIVE",
            thermal_status="NORMAL",
            device_profile_name="laptop"
        )

    def update_state(self, telemetry: Dict[str, Any], temp_data: Dict[str, Any], profile_name: str) -> SystemBeliefState:
        now = time.time()
        dt = max(0.1, now - self.last_time)
        
        temp_c = temp_data.get("temperature_celsius", 25.0)
        cpu_p = telemetry.get("cpu_percent", 0.0)
        ram_p = telemetry.get("ram_percent", 0.0)
        battery_p = telemetry.get("battery_percent", 100.0)
        is_charging = telemetry.get("is_charging", True)

        # On initial pass, set smooth velocity
        if not hasattr(self, "_initialized") or not self._initialized:
            self._initialized = True
            velocity = 0.0
            acceleration = 0.0
        else:
            # Calculate thermal velocity (dT/dt)
            velocity = (temp_c - self.last_temp) / dt
            # Calculate thermal acceleration (d2T/dt2)
            acceleration = (velocity - self.last_velocity) / dt

        self.last_temp = temp_c
        self.last_velocity = velocity
        self.last_time = now

        # Determine state classification
        if temp_c >= self.critical_temp + 5.0:
            status = "OVERHEAT"
        elif temp_c >= self.critical_temp:
            status = "CRITICAL"
        elif temp_c >= self.warning_temp:
            status = "WARM"
        else:
            status = "NORMAL"

        self.current_state = SystemBeliefState(
            timestamp=now,
            temperature_c=temp_c,
            temp_velocity_c_per_sec=round(velocity, 3),
            temp_acceleration=round(acceleration, 3),
            cpu_percent=cpu_p,
            ram_percent=ram_p,
            battery_percent=battery_p,
            is_charging=is_charging,
            current_action=self.current_state.current_action,
            thermal_status=status,
            device_profile_name=profile_name
        )
        return self.current_state

    def predict_future_state(self, candidate_action: str, horizon_sec: float = 5.0) -> Dict[str, float]:
        """
        Transition Model: Predicts expected temperature T(t+h) and CPU load under candidate_action.
        """
        state = self.current_state
        base_temp = state.temperature_c
        velocity = state.temp_velocity_c_per_sec

        # Action thermal modifier
        # Action choices: PASSIVE, COOL_MODERATE, THROTTLE_LIGHT, THROTTLE_HEAVY, EMERGENCY_COOLING, REDUCE_PAYLOAD, SLEEP_DUTY_CYCLE
        cooling_power = 0.0
        cpu_load_reduction = 0.0

        if candidate_action == "COOL_MODERATE":
            cooling_power = 0.8
        elif candidate_action == "THROTTLE_LIGHT":
            cooling_power = 1.2
            cpu_load_reduction = 15.0
        elif candidate_action == "THROTTLE_HEAVY":
            cooling_power = 2.5
            cpu_load_reduction = 40.0
        elif candidate_action == "EMERGENCY_COOLING" or candidate_action == "EMERGENCY_LANDING_ALERT":
            cooling_power = 4.0
            cpu_load_reduction = 60.0
        elif candidate_action == "REDUCE_PAYLOAD":
            cooling_power = 1.8
            cpu_load_reduction = 30.0
        elif candidate_action == "SLEEP_DUTY_CYCLE":
            cooling_power = 3.0
            cpu_load_reduction = 70.0

        # Forecasted temperature based on thermal physics model prediction
        predicted_velocity = velocity - (cooling_power * 0.15)
        predicted_temp = base_temp + (predicted_velocity * horizon_sec)
        predicted_cpu = max(5.0, state.cpu_percent - cpu_load_reduction)

        return {
            "predicted_temp_c": round(max(20.0, predicted_temp), 2),
            "predicted_velocity": round(predicted_velocity, 3),
            "predicted_cpu_percent": round(predicted_cpu, 1)
        }
