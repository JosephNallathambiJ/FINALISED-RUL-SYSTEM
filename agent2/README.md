# Agent 2 - Fan RPM & Active Cooling AI Agent

An agentic AI system created to explicitly monitor and control **CPU Fan RPM, PWM Duty Cycles, Acoustic Noise (dBA), Airflow (CFM), and Thermal Dissipation** across **Laptops, Desktop Workstations, Drones, and IoT Edge Devices**.

---

## 🧠 Core Architecture & Features

1. **Closed-Loop PID Fan Speed Controller**: Precision Proportional-Integral-Derivative ($K_p, K_i, K_d$) loop with anti-windup clamping to maintain target CPU setpoints without thermal oscillation.
2. **Model-Based Cooling Dynamics**: Computes airflow CFM output, acoustic noise level (dBA), and forecasts temperature $T_{next}(\text{RPM})$.
3. **Utility-Based Trade-off Engine**: Evaluates candidate RPM setpoints balancing **Thermal Safety**, **Acoustic Noise (dBA)**, **Power Consumption**, and **RPM Flapping/Hunting penalties**.
4. **Learning Agent & Fan Health Monitor**: Logs past history in SQLite (`rpm_cooling_history.db`) and detects fan degradation, mechanical friction, or stall conditions (PWM > 25% but 0 RPM).
5. **Hardware Ready**: Reads Linux hwmon fan tachometers (`/sys/class/hwmon/hwmon*/fan*_input`) and provides physical hardware callbacks for RPM reading and PWM actuation.

---

## 🔌 Attaching Physical Fan Hardware (Tachometer & PWM)

```python
from agent2_core.agent import FanCoolingAgent

agent = FanCoolingAgent()

# 1. Attach physical Fan Tachometer RPM sensor function
def read_physical_fan_rpm() -> float:
    # Example reading from GPIO / pulse counter pin / Arduino serial
    return 2850.0

agent.register_rpm_sensor_callback(read_physical_fan_rpm)

# 2. Attach physical PWM hardware control output function
def write_physical_pwm(duty_cycle_pct: float):
    # Example writing duty cycle to physical PWM pin
    print(f"Setting physical PWM hardware output to {duty_cycle_pct}%")

agent.register_pwm_actuator_callback(write_physical_pwm)

# 3. Step agent loop
agent.step()
```

---

## 🚀 Usage & Commands

### 1. Automated System Verification Test

```bash
cd agent2
python3 main.py --test-run
```

### 2. Hardware Tachometer & PWM Hook Test

```bash
python3 main.py --rpm-test
```

### 3. Run Autonomous CLI Monitoring Loop

```bash
# Laptop profile (Silent acoustics priority)
python3 main.py --profile laptop

# Desktop Workstation profile (High cooling performance)
python3 main.py --profile computer

# Drone ESC cooling motor profile (Zero noise penalty, high speed)
python3 main.py --profile drone
```

### 4. Launch Interactive Web Monitor Dashboard

```bash
python3 main.py --web --port 8001
```

Open **`http://localhost:8001`** in your browser to view the real-time glassmorphic monitor featuring live RPM speed gauges, acoustic noise meters (dBA), PID terms breakdown, and thermal spike load testing.
