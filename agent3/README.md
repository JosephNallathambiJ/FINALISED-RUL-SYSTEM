# Agent 3 - System Voltage & Electrical Power AI Agent

An agentic AI system created to explicitly monitor and regulate **System Voltages, Multi-Rail Tolerances (VCore, 12V, 5V, 3.3V, Battery), Voltage Sags/Surges, and Power Quality** across **Laptops, Desktop Computers, Drones, and IoT Edge Devices**.

---

## 🧠 Core Architecture & Features

1. **Model-Based Electrical Dynamics**: Tracks multi-rail belief state, calculates total Power Consumption in Watts ($P = V \cdot I$), and models voltage droop under load ($\Delta V = I \cdot R_{internal}$).
2. **Utility-Based Voltage Protection**: Multi-attribute utility $U(S, A_{power})$ balancing **Voltage Stability ($\pm 5\%$)**, **Brownout Crash Prevention**, **Overvoltage Surge Guard**, and **Energy Efficiency**.
3. **Learning Agent & Transient Monitor**: Persists time-series voltage data in SQLite (`voltage_history.db`), learning baseline voltage ripple and detecting voltage sags, spikes, and brownout risks.
4. **Hardware Ready**: Reads Linux hwmon voltage inputs (`/sys/class/hwmon/hwmon*/in*_input`) and provides hardware callbacks for physical voltage ADCs (INA219, ADS1115, Arduino) and power actuators.

---

## 🔌 Attaching Physical Voltage Hardware (ADCs & Relays)

```python
from agent3_core.agent import SystemVoltageAgent

agent = SystemVoltageAgent()

# 1. Attach physical Voltage ADC reading function (e.g. INA219 / ADS1115 I2C)
def read_physical_voltage_adc() -> dict:
    return {
        "vcore_volts": 1.152,
        "v12_volts": 12.04,
        "v5_volts": 5.01,
        "v33_volts": 3.30,
        "battery_volts": 11.40,
        "power_watts": 24.8
    }

agent.register_voltage_sensor_callback(read_physical_voltage_adc)

# 2. Attach physical Power VRM / Relay Actuator function
def write_physical_power_vrm(action: str, vcore_delta: float):
    print(f"VRM Command: Executing {action} with offset {vcore_delta:+.3f}V")

agent.register_power_actuator_callback(write_physical_power_vrm)

# 3. Step agent loop
agent.step()
```

---

## 🚀 Usage & Commands

### 1. Automated System Verification Test

```bash
cd agent3
python3 main.py --test-run
```

### 2. Hardware Voltage Sensor ADC & Power Actuator Test

```bash
python3 main.py --voltage-test
```

### 3. Run Autonomous CLI Voltage Loop

```bash
# Laptop profile (VCore & Battery monitoring)
python3 main.py --profile laptop

# Desktop Workstation profile (Multi-Rail ATX +12V/+5V/+3.3V)
python3 main.py --profile computer

# Drone LiPo Battery profile (Flight battery cell safety)
python3 main.py --profile drone
```

### 4. Launch Real-Time Web Oscilloscope Dashboard

```bash
python3 main.py --web --port 8002
```

Open **`http://localhost:8002`** in your browser to view the real-time glassmorphic monitor featuring live Voltage Oscilloscope Scope Waveforms, Multi-Rail Status Gauges, and Voltage Sag/Surge Transient testing.
