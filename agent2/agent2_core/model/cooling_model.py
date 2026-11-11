import math
import time
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FanStateBelief:
    timestamp: float
    current_rpm: int
    target_pwm_pct: float
    temperature_c: float
    temp_velocity: float
    cpu_percent: float
    airflow_cfm: float
    acoustic_noise_dba: float
    thermal_status: str
    fan_health_status: str  # "OK", "STALL_WARNING", "DEGRADED"

class CoolingDynamicsModel:
    """
    Model-Based Agent Subsystem for Active Cooling & Fan Dynamics:
    1. Tracks current state of Fan RPM, Airflow (CFM), Acoustic Noise (dBA)
    2. Transition Model: Predicts expected next temperature T(t+h) under candidate Fan RPM setpoints.
    """

    def __init__(self, max_rpm: int = 5000, target_temp: float = 65.0, max_noise_dba: float = 45.0):
        self.max_rpm = max_rpm
        self.target_temp = target_temp
        self.max_noise_dba = max_noise_dba
        self.last_temp = 35.0
        self.last_time = time.time()
        self.simulated_temp = 42.0

    def compute_airflow_cfm(self, rpm: int) -> float:
        """Calculates cooling airflow in Cubic Feet per Minute (CFM)."""
        ratio = max(0.0, min(1.0, rpm / max(1, self.max_rpm)))
        max_cfm = 28.0  # Laptop/Workstation fan typical max CFM
        return round(max_cfm * (ratio ** 1.1), 2)

    def compute_acoustic_noise_dba(self, rpm: int) -> float:
        """
        Calculates fan motor & blade aerodynamic noise level in dBA.
        Ambient quiet room baseline ~ 25 dBA.
        """
        if rpm <= 100:
            return 22.0  # Silent fan off
        ratio = rpm / max(1, self.max_rpm)
        # Logarithmic noise scaling curve
        noise_dba = 24.0 + (32.0 * math.pow(ratio, 1.4))
        return round(min(85.0, noise_dba), 1)

    def predict_temperature_at_rpm(self, current_temp: float, cpu_load: float, target_rpm: int, horizon_sec: float = 3.0) -> float:
        """
        Forecasts expected temperature T(t+h) under candidate fan RPM setpoint.
        """
        airflow_cfm = self.compute_airflow_cfm(target_rpm)
        ambient_temp = 25.0
        
        # Heat generation rate from CPU load
        heat_gen = (cpu_load / 100.0) * 0.40 * 12.0
        # Heat dissipation rate from fan airflow & thermal delta
        cooling_loss = (current_temp - ambient_temp) * 0.04 + (airflow_cfm * 0.18)
        
        predicted_velocity = heat_gen - cooling_loss
        predicted_temp = current_temp + (predicted_velocity * horizon_sec)
        return round(max(ambient_temp, predicted_temp), 2)

    def update_belief_state(self, current_rpm: int, pwm_pct: float, current_temp: float, cpu_p: float) -> FanStateBelief:
        now = time.time()
        dt = max(0.1, now - self.last_time)
        temp_vel = (current_temp - self.last_temp) / dt
        self.last_temp = current_temp
        self.last_time = now

        cfm = self.compute_airflow_cfm(current_rpm)
        dba = self.compute_acoustic_noise_dba(current_rpm)

        # Thermal status
        if current_temp >= 80.0:
            t_status = "CRITICAL"
        elif current_temp >= 68.0:
            t_status = "WARM"
        else:
            t_status = "NORMAL"

        # Fan health status check
        health = "OK"
        if pwm_pct > 30.0 and current_rpm < 100:
            health = "STALL_WARNING" # PWM command active but 0 RPM spinning!

        return FanStateBelief(
            timestamp=now,
            current_rpm=current_rpm,
            target_pwm_pct=pwm_pct,
            temperature_c=current_temp,
            temp_velocity=round(temp_vel, 2),
            cpu_percent=cpu_p,
            airflow_cfm=cfm,
            acoustic_noise_dba=dba,
            thermal_status=t_status,
            fan_health_status=health
        )
