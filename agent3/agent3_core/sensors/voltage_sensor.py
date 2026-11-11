import psutil
import time
import glob
import random
from typing import Dict, Any, Callable, Optional
from .base_sensor import BaseSensor

class VoltageSensor(BaseSensor):
    """
    Multi-Rail Voltage Sensor Subsystem:
    1. Reads OS/sysfs hwmon voltage inputs (`/sys/class/hwmon/hwmon*/in*_input`)
    2. Supports Custom Hardware ADC Voltage Sensor Callbacks (INA219, ADS1115, Arduino)
    3. Dynamic Electrical Physics Simulator modeling VCore voltage droop & battery drain
    """

    def __init__(self, mode: str = "auto", vcore_nom: float = 1.15, v12_nom: float = 12.0, battery_nom: float = 11.4):
        self.mode = mode.lower()
        self.vcore_nom = vcore_nom
        self.v12_nom = v12_nom
        self.battery_nom = battery_nom
        self.custom_voltage_callback: Optional[Callable[[], Dict[str, float]]] = None
        self.simulated_battery_volts = battery_nom if battery_nom > 0 else 12.0
        self._last_time = time.time()

    def register_voltage_sensor(self, callback_fn: Callable[[], Dict[str, float]]):
        """
        User hook to attach a physical voltage ADC / Sensor!
        Example:
            sensor.register_voltage_sensor(lambda: {"vcore": my_ina219.read_bus_voltage()})
        """
        self.custom_voltage_callback = callback_fn
        self.mode = "custom"

    def read_hardware_voltage_sysfs(self) -> Optional[Dict[str, float]]:
        """Attempts to read actual voltages from Linux sysfs hwmon in*_input."""
        try:
            in_files = glob.glob("/sys/class/hwmon/hwmon*/in*_input")
            if in_files:
                rail_map = {}
                for idx, fpath in enumerate(in_files[:4]):
                    with open(fpath, "r") as f:
                        val = f.read().strip()
                        if val.isdigit():
                            volts = int(val) / 1000.0
                            rail_map[f"rail_{idx}"] = round(volts, 3)
                if rail_map:
                    return rail_map
        except Exception:
            pass
        return None

    def simulate_electrical_physics(self, cpu_load: float) -> Dict[str, float]:
        """
        Physics-based electrical simulation:
        VCore droop: V = V_nominal - (Load_ratio * Droop_R) + Noise
        """
        now = time.time()
        dt = max(0.1, min(now - self._last_time, 2.0))
        self._last_time = now

        # VCore voltage droop under load
        load_ratio = cpu_load / 100.0
        r_internal = 0.08  # Ohms equivalent droop resistance
        current_amps = 15.0 + (load_ratio * 45.0)  # Simulated CPU current draw
        
        vcore_droop = current_amps * (r_internal * 0.005)
        vcore_volts = self.vcore_nom - vcore_droop + random.uniform(-0.008, 0.008)

        # 12V Rail (ATX / Drone main supply)
        v12_volts = self.v12_nom - (load_ratio * 0.12) + random.uniform(-0.02, 0.02) if self.v12_nom > 0 else 0.0

        # 5V Rail
        v5_volts = 5.00 + random.uniform(-0.03, 0.03)

        # 3.3V Rail
        v33_volts = 3.30 + random.uniform(-0.02, 0.02)

        # Battery discharge simulation
        if self.battery_nom > 0:
            discharge_rate = (0.002 + (load_ratio * 0.008)) * dt
            self.simulated_battery_volts = max(8.5, self.simulated_battery_volts - discharge_rate)
            battery_v = self.simulated_battery_volts
        else:
            battery_v = 0.0

        return {
            "vcore_volts": round(vcore_volts, 3),
            "v12_volts": round(v12_volts, 2),
            "v5_volts": round(v5_volts, 2),
            "v33_volts": round(v33_volts, 2),
            "battery_volts": round(battery_v, 2),
            "current_amps": round(current_amps, 1),
            "power_watts": round(vcore_volts * current_amps, 1)
        }

    def read(self, cpu_load: float = 0.0) -> Dict[str, Any]:
        rail_data = {}
        source_name = "simulated_electrical_physics"

        if self.mode == "custom" and self.custom_voltage_callback is not None:
            try:
                rail_data = self.custom_voltage_callback()
                source_name = "custom_hardware_adc_sensor"
            except Exception:
                rail_data = self.simulate_electrical_physics(cpu_load)
                source_name = "custom_error_fallback"

        elif self.mode in ("auto", "hardware"):
            hw_volts = self.read_hardware_voltage_sysfs()
            if hw_volts is not None:
                rail_data = hw_volts
                source_name = "os_hwmon_sysfs"
            else:
                rail_data = self.simulate_electrical_physics(cpu_load)
                source_name = "simulated_electrical_physics"
        else:
            rail_data = self.simulate_electrical_physics(cpu_load)
            source_name = "simulated_electrical_physics"

        return {
            "timestamp": time.time(),
            "rails": rail_data,
            "sensor_source": source_name,
            "is_custom_attached": (self.custom_voltage_callback is not None)
        }
