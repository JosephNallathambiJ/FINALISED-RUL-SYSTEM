# Agent1 - Model-Based, Utility-Based, Learning AI Agent

An agentic AI system created to monitor CPUs and IoT systems across diverse hardware profiles (**Laptops, Desktop Computers, Drones, and IoT Edge Devices**).

---

## 🧠 AI Agent Classification

This agent strictly satisfies three core classical AI agent paradigms (Russell & Norvig model):

1. **Model-Based Agent**: Maintains an internal **Belief State** $S_t = \langle T_t, \Delta T_t, \Delta^2 T_t, \text{CPU}_t, \text{Battery}_t \rangle$ and physics transition model $P(S_{t+1} \mid S_t, A)$ to forecast future thermal dynamics under candidate actions.
2. **Utility-Based Agent**: Evaluates competing actions using a multi-attribute utility function $U(S, A)$ balancing thermal safety, compute throughput, power conservation, and action stability.
3. **Learning Agent**: Stores past history of all temperature readings, CPU load, and taken actions in a persistent time-series SQLite database. Performs online learning to establish dynamic baselines ($\mu, \sigma$), dynamic anomaly detection, and adaptive heating rate parameters.

---

## 🛸 Device Hardware Profiles

Configured via `config.yaml` or dynamically via CLI/Web API:

- **Drone (`--profile drone`)**: Optimized for autonomous UAVs. Prioritizes battery preservation and catastrophic thermal shutdown prevention in mid-air (Payload task shedding, Emergency Landing Alerts).
- **Laptop (`--profile laptop`)**: Balances active fan cooling, battery efficiency, and compute performance for portable computers.
- **Computer (`--profile computer`)**: Optimized for high-throughput desktop workstations with high thermal thresholds and max performance preference.
- **IoT Edge (`--profile iot`)**: Optimized for fanless, low-power edge nodes. Uses sleep duty cycles and low power consumption states.

---

## 🌡️ Attaching Physical Temperature Sensors

The agent features a plug-and-play callback hook so you can easily attach **any physical hardware temperature sensor** (e.g. Raspberry Pi I2C MCP9808, DS18B20 1-Wire, DHT22, Arduino Serial probe, etc.).

### Example: Attaching a Custom Sensor in Python

```python
from agent1_core.agent import ModelUtilityLearningAgent

# 1. Initialize Agent
agent = ModelUtilityLearningAgent()

# 2. Define your physical hardware temperature reading function
def read_my_i2c_sensor() -> float:
    # Read from your physical I2C / SPI / GPIO / Serial hardware
    import smbus
    bus = smbus.SMBus(1)
    raw_data = bus.read_i2c_block_data(0x18, 0x05, 2)
    # Convert hardware bytes to Celsius
    celsius = (raw_data[0] & 0x1F) * 16 + raw_data[1] / 16.0
    return celsius

# 3. Register your physical sensor with the agent
agent.register_hardware_temperature_sensor(read_my_i2c_sensor)

# 4. Run agent
agent.step()
```

---

## 🚀 Quickstart & Usage

### 1. Installation

Ensure Python 3.9+ is installed, then install requirements:

```bash
cd agent1
pip install -r requirements.txt
```

### 2. Run Automated Verification Test

```bash
python main.py --test-run
```

### 3. Run Hardware Temperature Sensor Test

```bash
python main.py --sensor-test
```

### 4. Run Autonomous CLI Monitoring Loop

Run agent targeting a specific hardware profile:

```bash
# Drone profile
python main.py --profile drone

# Laptop profile
python main.py --profile laptop

# IoT Edge profile
python main.py --profile iot
```

### 5. Launch Real-Time Web Monitor & API Dashboard

```bash
python main.py --web
```

Open your web browser at **`http://localhost:8000`** to access the interactive glassmorphic dashboard featuring real-time temperature charts, belief state indicators, utility score breakdown, and profile switching.

---

## 📊 Database & Temperature History

All recorded telemetry, past temperatures, utility scores, and action outputs are stored in `temperature_history.db` (SQLite).

Query temperature history via Python:

```python
from agent1_core.learning.history_store import TemperatureHistoryStore

db = TemperatureHistoryStore("temperature_history.db")
print(db.get_summary_stats())
recent = db.get_recent_history(limit=50)
```
