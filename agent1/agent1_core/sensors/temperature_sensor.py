import psutil
import time
import random
from typing import Dict, Any, Callable, Optional
from .base_sensor import BaseSensor

class TemperatureSensor(BaseSensor):
    """
    Temperature Sensor Module with support for:
    1. OS Hardware Thermal Probes (psutil/sysfs)
    2. User Custom Physical Hardware Sensor Callbacks (I2C/SPI/Arduino/Raspberry Pi)
    3. Realistic Thermal Dynamic Simulation Probe
    """

    def __init__(self, mode: str = "auto", ambient_temp: float = 28.0):
        self.mode = mode.lower()  # "auto", "hardware", "custom", "simulated"
        self.ambient_temp = ambient_temp
        self.simulated_temp = ambient_temp + 10.0
        self.custom_sensor_callback: Optional[Callable[[], float]] = None
        self._last_time = time.time()

    def register_custom_sensor(self, callback_fn: Callable[[], float]):
        """
        User hook to attach a custom physical temperature sensor!
        Example:
            sensor.register_custom_sensor(lambda: my_i2c_sensor.read_celsius())
        """
        self.custom_sensor_callback = callback_fn
        self.mode = "custom"

    def read_hardware_temperature(self) -> Optional[float]:
        """Attempts to read actual CPU/System hardware core temperature sensors."""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                all_temps = []
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current and entry.current > 0:
                            all_temps.append(entry.current)
                if all_temps:
                    return max(all_temps)  # Return peak core temp
        except Exception:
            pass
        return None

    def simulate_thermal_step(self, current_cpu_load: float, action_cooling: float = 0.0) -> float:
        """
        Physics-based thermal dynamic model:
        dT/dt = k_heat * cpu_load - k_cool * (T - T_ambient) - fan_cooling
        """
        now = time.time()
        dt = max(0.1, min(now - self._last_time, 5.0))
        self._last_time = now

        k_heat = 0.45          # Heat generation rate per % CPU load
        k_dissipation = 0.08   # Passive dissipation rate towards ambient
        k_fan = 0.35 * action_cooling # Active cooling effect

        heat_gen = (current_cpu_load / 100.0) * k_heat * 15.0
        heat_loss = (self.simulated_temp - self.ambient_temp) * k_dissipation + k_fan * 10.0
        
        # Thermal inertia update
        self.simulated_temp += (heat_gen - heat_loss) * dt
        # Add slight realistic thermal noise
        self.simulated_temp += random.uniform(-0.15, 0.15)
        # Thermal floor clamp
        self.simulated_temp = max(self.ambient_temp, round(self.simulated_temp, 2))
        return self.simulated_temp

    def read(self, current_cpu_load: float = 0.0, action_cooling: float = 0.0) -> Dict[str, Any]:
        temp_val: Optional[float] = None
        source_name = "simulated"

        if self.mode == "custom" and self.custom_sensor_callback is not None:
            try:
                temp_val = float(self.custom_sensor_callback())
                source_name = "custom_hardware_sensor"
            except Exception as e:
                # Fallback on custom error
                source_name = f"custom_error_fallback"
                temp_val = self.simulate_thermal_step(current_cpu_load, action_cooling)

        elif self.mode in ("auto", "hardware"):
            temp_val = self.read_hardware_temperature()
            if temp_val is not None:
                source_name = "os_hardware_sensor"
            else:
                source_name = "simulated_thermal_dynamics"
                temp_val = self.simulate_thermal_step(current_cpu_load, action_cooling)
        else:
            source_name = "simulated_thermal_dynamics"
            temp_val = self.simulate_thermal_step(current_cpu_load, action_cooling)

        return {
            "timestamp": time.time(),
            "temperature_celsius": round(temp_val, 2),
            "sensor_source": source_name,
            "is_custom_attached": (self.custom_sensor_callback is not None)
        }
