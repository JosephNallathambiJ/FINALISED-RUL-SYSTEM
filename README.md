# Machine Learning Framework for Physical System Failure Prediction and Reliability Assessment

IQAC is a production-grade, two-computer predictive maintenance and prognostics platform designed for CPUs, IoT edge gateways, enterprise servers, and embedded devices. It integrates hardware operating-system telemetry, specialized autonomous decision agents, cumulative physics-informed degradation modeling, machine learning regression, and an agentic AI diagnostic layer.

---

## What the System Does

1. **Hardware Telemetry Collection (Computer 1 - Monitored Device):**
   - Continuously samples CPU core utilization, temperatures (via OS probes, Linux hwmon, or external DS18B20 1-Wire sensors), core voltage rails, fan RPM, CPU clock frequency, and thermal throttling events.
   - Accurately tracks accumulated operational uptime without sleep or reboot inflation.
   - Rejects zero, negative, or corrupted sensor spikes without polluting daily moving averages.

2. **Autonomous Decision Agents (Agents 1 through 4):**
   - **Agent 1:** Model-based, utility-based thermal monitoring with learning baselines.
   - **Agent 2:** Active cooling with PID fan speed controller and thermal dynamic forecasting.
   - **Agent 3:** Multi-rail electrical power and voltage protection (VCore, 12V, 5V, 3.3V).
   - **Agent 4:** CPU clock frequency, governor mode, and dynamic performance management.

3. **Scientifically Defensible RUL Engine (Computer 2 - Prediction Machine):**
   - Implements **Arrhenius thermal acceleration**, **Coffin-Manson thermal cycling fatigue**, and **Black's law electromigration & TDDB field acceleration**.
   - Integrates cumulative wear using **Miner's rule** across discrete observation intervals (satisfying Jensen's inequality for fluctuating thermal workloads).
   - Operates in three distinct prognostic modes:
     - **MODE 1: Physics-Based Estimation**
     - **MODE 2: Supervised Machine Learning Prediction** (Random Forest & Gradient Boosting/XGBoost)
     - **MODE 3: Hybrid Physics-Informed ML** (Bayesian fusion with explicit model disagreement and uncertainty envelopes)
   - Features component-specific reliability profiles (`desktop`, `server`, `laptop`, `iot`).

4. **Agentic AI Diagnostics & Decision Support (Computer 2):**
   - Interfaces with local Ollama models (optimized for AI Cyber Architecture models such as Dolphin3-Cyber or Gemma) to analyze telemetry trends, hypothesize physical failure modes (thermal paste degradation, fan bearing wear, heatsink blockage), and generate prioritized mitigation steps.
   - Guarantees zero crashes with a deterministic expert synthesizer fallback when local LLMs are offline.

5. **Interactive Prognostics Dashboard:**
   - Serves a modern glassmorphic web dashboard displaying live KPIs, historical excursion charts, cumulative degradation curves, 90% confidence bands, and AI diagnostic reports.

---

## Two-Computer Architecture and Data Flow

```text
┌─────────────────────────────────────────────────────────────┐
│             COMPUTER 1: MONITORED DEVICE (PORT 8001)        │
│                                                             │
│   OS Sensors / hwmon / 1-Wire DS18B20                       │
│                       │                                     │
│                       ▼                                     │
│   Agents 1-4 (Thermal, Fan PID, Voltage, Governor)          │
│                       │                                     │
│                       ▼                                     │
│   recorder.py (Input Validation, Uptime Accumulation)       │
│                       │                                     │
│                       ▼                                     │
│   monitoring_records.db (Daily Summaries & Spikes)          │
│                       │                                     │
│                       ▼                                     │
│   FastAPI Server (GET /records, GET /)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Wi-Fi / Local Network Sync (HTTP GET /records)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          COMPUTER 2: PREDICTION MACHINE (PORT 8000)         │
│                                                             │
│   sync.py / monitor_client.py (Exponential Backoff, Auth)   │
│                       │                                     │
│                       ▼                                     │
│   database.py -> telemetry.db (Indexed Metrics & Intervals) │
│                       │                                     │
│                       ▼                                     │
│   Prognostic Engine (rul_engine.py & ml_engine.py)          │
│     * Mode 1: Arrhenius / Coffin-Manson / Miner's Rule      │
│     * Mode 2: Random Forest / XGBoost Regressor             │
│     * Mode 3: Uncertainty-Weighted Hybrid Fusion            │
│                       │                                     │
│                       ▼                                     │
│   Agentic AI Layer (agent.py -> Ollama Cyber / Gemma)       │
│                       │                                     │
│                       ▼                                     │
│   FastAPI Prognostic API & Glassmorphic Dashboard           │
│   (GET /devices/{id}/rul, GET /dashboard, GET /history)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

| Path | Component | Purpose |
| --- | --- | --- |
| `overall_run.py` | Single-Click Entrypoint | Master orchestrator with dependency check, role selection, subprocess tracking, and auto-browser launcher. |
| `control/server_manager.py` | Server Management | Lifecycle supervision (Start, Stop, Restart, Status, Test Connection), LAN/VPN binding, client registry. |
| `control/pairing_manager.py` | Cryptographic Pairing | Expiring single-use pairing codes (`PAIR-XXXXXX`), PBKDF2-HMAC tokens, persistent device registry. |
| `control/authentication.py` | Security Engine | PBKDF2-HMAC-SHA256 password hashing, sliding-window rate limiting, session and host validation. |
| `control/services.py` | Subprocess Manager | Port conflict detection, background process supervision, circular log buffer, controlled stress test. |
| `control/connections.py` | Network Architecture | Interface detection (IPv4, Wi-Fi, Ethernet, VPN), hardware discovery, and peer ping latency testing. |
| `control/measurements.py` | Telemetry Engine | High-frequency hardware telemetry sampler with circular ring-buffer metrics. |
| `dashboard/app.py` | Enterprise Web Server | FastAPI application serving static assets, REST APIs, and Origin-protected WebSockets. |
| `dashboard/routes.py` | Enterprise REST API | Server management, cryptographic pairing, telemetry series, thermal spikes, hardware inventory endpoints. |
| `dashboard/websocket.py` | Real-time Stream | Low-latency telemetry broadcasting with client heartbeat tracking and automatic cleanup. |
| `dashboard/templates/index.html`| Industrial UI Shell | Enterprise SCADA/observability dashboard with collapsible 5-group sidebar and 16 dedicated views. |
| `dashboard/static/style.css` | Design System | Dark graphite (`#090d16`), navy slate surfaces, cyan accents, responsive table & card grids. |
| `dashboard/static/app.js` | UI Controller | Chart.js telemetry charts, time filtering (1m, 5m, 30m, 1h), modal dialogs, and API client. |
| `run.py` | Monitoring Side | Production entry point starting monitor loop and records API together. |
| `recorder.py` | Monitoring Side | Validates sensor data, prevents zero-dilution, tracks uptime, serves `/records`. |
| `simulation.py` | Monitoring Side | Smoke-test loop executing agents with unified cross-agent sensor recording. |
| `hardware_monitor.py` | Monitoring Side | Raspberry Pi / Linux 1-Wire DS18B20 physical hardware monitor. |
| `spike.py` | Monitoring Side | Standalone threshold spike watcher for CPU, memory, temp, and battery. |
| `iqac-prediction/rul_engine.py` | Prediction Engine | Multi-mode physics degradation and Arrhenius cumulative damage engine. |
| `iqac-prediction/ml_engine.py` | Prediction Engine | Supervised ML regression, prognostic metrics, and hybrid fusion strategy. |
| `iqac-prediction/train.py` | ML Pipeline | Time-aware training pipeline for Random Forest and XGBoost/GB. |
| `iqac-prediction/database.py` | Prediction Storage | SQLite persistence, schema migrations, and validated metric aggregation. |
| `iqac-prediction/monitor_client.py` | Synchronization | Telemetry normalization and client with exponential backoff retries. |
| `iqac-prediction/sync.py` | Synchronization | CLI tool for synchronizing remote monitoring records. |
| `iqac-prediction/api.py` | Prognostic API | Structured RUL endpoints, history, and dashboard server. |
| `iqac-prediction/agent.py` | Agentic AI | Ollama LLM diagnostic agent with deterministic tool calling and expert fallback. |
| `tests/` | Automated Test Suite | Comprehensive unit, integration, and synthetic tests (55 passing tests). |

---

## Mathematical Formulation of the RUL Engine

### 1. Temperature Acceleration (Arrhenius Relationship)
Thermal acceleration of chemical and physical reaction rates (electromigration, dielectric breakdown) follows:

$$\text{AF}_{\text{Arrhenius}}(T) = \exp\left[\frac{E_a}{k_B} \left(\frac{1}{T_{\text{ref, K}}} - \frac{1}{T_{\text{use, K}}}\right)\right]$$

where:
- $E_a$ = Activation energy ($0.65\text{--}0.75\text{ eV}$ depending on CPU component profile).
- $k_B$ = Boltzmann constant ($8.61733 \times 10^{-5}\text{ eV/K}$).
- $T_{\text{ref}}$ = Component reference baseline temperature (Kelvin).
- $T_{\text{use}}$ = Measured silicon operating temperature (Kelvin).

### 2. Voltage Acceleration (Black's Equation & TDDB)
Electric field stress accelerates gate oxide breakdown and electromigration nonlinearly:

$$\text{AF}_{\text{Voltage}}(V) = \left(\frac{V}{V_{\text{ref}}}\right)^n$$

where $n \approx 2.0$, and $V_{\text{ref}}$ is nominal VCore voltage (e.g. $1.20\text{V}$).

### 3. Thermal Cycling Fatigue (Coffin-Manson Model)
Repeated rapid thermal expansions and contractions ($\Delta T$) cause mechanical solder joint fatigue and silicon interposer micro-fractures:

$$\text{Damage}_{\text{cycle}} = N_{\text{spikes}} \times C \cdot \left(\frac{\Delta T}{\Delta T_{\text{ref}}}\right)^q$$

where $q = 2.5$ and $\Delta T$ is temperature excursion magnitude above nominal.

### 4. Cumulative Damage Integration (Miner's Rule)
To avoid underestimating wear caused by thermal transients (due to Jensen's inequality: $\mathbb{E}[e^T] \ge e^{\mathbb{E}[T]}$), damage is integrated cumulatively across discrete observation intervals:

$$D(t) = \sum_{k=1}^M \frac{\Delta t_k \cdot \text{AF}_{\text{Arrhenius}}(T_k) \cdot \text{AF}_{\text{Voltage}}(V_k)}{L_0} + \text{Damage}_{\text{cycle}}$$

where $L_0$ is rated baseline lifespan at reference conditions.

### 5. Prognostic Health & RUL Calculation
$$\text{Health}(t) = \max\left(0, 100 \times [1.0 - D(t)]\right)$$

$$\text{RUL}_{\text{physics}}(t) = \begin{cases} \frac{1.0 - D(t)}{\bar{r}_D} & \text{if } D(t) < 1.0 \text{ and } \bar{r}_D > 0 \\ 0.0 & \text{if } D(t) \ge 1.0 \end{cases}$$

---

## Machine Learning & Hybrid Prediction

- **Supervised ML Regressors:** Implements Random Forest Regressors and Gradient Boosting/XGBoost on physics-informed temporal features (rolling temperatures, variance, Arrhenius stress index, spike count, voltage).
- **Time-Aware Validation:** Uses strict chronological train/test splitting to prevent future-to-past data leakage.
- **Prognostic Metrics:** Evaluates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and the asymmetric **NASA PHM Score** which penalizes dangerous late predictions exponentially harder than safe early predictions:

$$S = \sum_{i=1}^N \begin{cases} \exp\left(-\frac{\hat{y}_i - y_i}{13}\right) - 1 & \hat{y}_i < y_i \text{ (early / safe)} \\ \exp\left(\frac{\hat{y}_i - y_i}{10}\right) - 1 & \hat{y}_i \ge y_i \text{ (late / hazardous)} \end{cases}$$

- **Hybrid Fusion Strategy:** Blends physics estimates ($RUL_{\text{phys}}$) and empirical ML predictions ($RUL_{\text{ml}}$) using inverse-variance weights. If model disagreement $\delta = |RUL_{\text{phys}} - RUL_{\text{ml}}| / \max(RUL_{\text{phys}}, RUL_{\text{ml}}) > 0.40$, the system explicitly logs an elevation warning and expands the 90% confidence interval to span the full envelope $[\min(\text{lower}), \max(\text{upper})]$.
- **Refusal to Fabricate:** The system refuses to train on datasets lacking ground-truth failure targets unless explicitly directed via benchmark flags, and clearly annotates uncalibrated heuristic artifacts.

---

## Installation & Setup

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn xgboost requests pytest langchain langchain-community
```

---

## Single-Click Control Center (`overall_run.py`)

The system includes a single-click industrial launcher that eliminates the need to execute multiple Python scripts through separate terminal tabs.

```bash
source .venv/bin/activate
python overall_run.py
```

### CLI Parameters:
```text
usage: overall_run.py [-h] [--role {MONITORING,PREDICTION,LOCAL_DEMO}]
                      [--port PORT] [--host HOST] [--no-browser]
                      [--interactive]

Single-Click CPU Monitoring & RUL Prediction Control Center

optional arguments:
  --role {MONITORING,PREDICTION,LOCAL_DEMO}
                        Operational role for this machine
  --port PORT           Dashboard port (default: 8080)
  --host HOST           Network host interface (default: 0.0.0.0 for LAN access)
  --no-browser          Do not automatically open web browser
  --interactive         Prompt for role selection interactively
```

When launched, `overall_run.py`:
1. Detects host OS, CPU core counts, architecture, and network interfaces.
2. Validates dependencies and initialises SQLite database schemas (`telemetry.db`, `monitoring_records.db`, `paired_devices.json`).
3. Dispatches and supervises subprocesses according to the selected role.
4. Starts the enterprise dashboard on `0.0.0.0:8080` (accessible via localhost, LAN IP, or VPN).
5. Opens the browser automatically while providing a clean terminal banner with connection URLs.
6. Installs graceful `SIGINT`/`SIGTERM` handlers to terminate all background child processes upon exit.

---

## Enterprise Control Platform & UI Architecture

The redesigned industrial dashboard presents an enterprise observability interface featuring:
- **Design System:** Dark graphite (`#090d16`), navy/slate panels (`#141b29`), restrained cyan (`#00e5ff`) and electric blue (`#38bdf8`) accents, consistent typography, and Lucide icons (no gaming gradients or emojis).
- **Collapsible Industrial Sidebar:** Structured into 5 functional operational domains with 16 views:
  - **OVERVIEW:**
    - `System Overview`: High-level KPI summary, hardware specs, network state, and real-time degradation gauges.
    - `Live Monitoring`: High-frequency CPU metrics with selectable temporal filters (`1m`, `5m`, `30m`, `1h`, `All`).
    - `Hardware Inventory`: Core architecture, cache topology, clock stepping, and fan sensors.
  - **PREDICTIVE MAINTENANCE:**
    - `RUL Prediction`: Interactive Arrhenius, ML, and Hybrid RUL calculator with 90% confidence bands.
    - `Degradation Analysis`: Cumulative damage curves, Coffin-Manson fatigue cycles, and wear indices.
    - `Thermal Stress History`: Excursion duration tracking, high-temperature events, and thermal spikes table.
    - `Prediction History`: Audit trail of historical RUL estimates, health scores, and timestamps.
  - **INTELLIGENCE:**
    - `Agentic AI Diagnostics`: Autonomous Ollama AI Cyber / Gemma diagnosis with root-cause physical failure hypotheses.
    - `Diagnostic Reports`: Structured diagnostic dossiers with sensor evidence and priority actions.
    - `Maintenance Recommendations`: Actionable maintenance schedule (thermal compound repasting, heatsink decrowning, voltage trimming).
  - **INFRASTRUCTURE:**
    - `Device Management`: Paired node registry, device metadata, connection health, and remote telemetry links.
    - `Remote Connections`: Network ping latency monitor, remote API endpoint tester, and IP discovery.
    - `Server Management`: Full server lifecycle control (Start, Stop, Restart, Status, Port Binding, Connected Clients list with Disconnect controls).
    - `Measurement Control`: Live telemetry sampling rate controller, circular buffer inspection, and controlled CPU stress testing with 85°C auto-cutoff.
  - **SYSTEM:**
    - `Logs and Events`: Live circular log stream with severity filtering (`ALL`, `INFO`, `WARNING`, `ERROR`).
    - `Settings`: Operational parameters, Arrhenius reference baselines, and security configuration.

---

## Remote Server Hosting & Cross-Network Access

The platform natively supports remote hosting for data-center, factory-floor, and multi-computer deployments.

### 1. Local Network (LAN) Hosting
By default, the server binds to `0.0.0.0:8080`. Any device on the same local network or Wi-Fi subnet can access the control center:
```text
http://<YOUR_LAN_IP>:8080
```
*(Example: `http://192.168.1.105:8080`)*

The current LAN IPv4 address is automatically discovered and prominently displayed in the top navigation bar and the Server Management pane.

### 2. Secure Remote Hosting via VPN Mesh (Tailscale / WireGuard)
For remote access across different physical locations without exposing ports to the public Internet:
- Install **Tailscale** or **WireGuard** on both the server and client machines.
- The platform automatically detects VPN network interfaces (e.g., `tailscale0`, `utun*`, `wg0`) and displays the secure VPN address:
```text
http://100.x.y.z:8080
```
- Only authenticated devices on your private tailnet / mesh can connect.

### 3. Server Management REST API
The server lifecycle can be monitored and commanded programmatically:
- `GET /api/server/status`: Server uptime, active host, port, mode, and client count.
- `POST /api/server/start`: Start server instances.
- `POST /api/server/stop`: Gracefully stop dashboard and background services.
- `POST /api/server/restart`: Perform hot restart.
- `POST /api/server/test`: Test TCP reachability of target IP and port.
- `GET /api/server/clients`: List connected WebSocket and HTTP client IPs and active session durations.
- `POST /api/server/clients/disconnect`: Forcibly terminate a connected client session.

---

## Cryptographic Device Pairing Workflow

To prevent unauthorized devices from injecting forged telemetry or reading sensitive hardware diagnostics, the platform implements a cryptographic challenge-handshake pairing mechanism:

1. **Generate Single-Use Pairing Code:**
   - In the **Device Management** or **Remote Connections** tab, click **"Generate Pairing Code"**.
   - The server issues a secure, single-use, 6-character alphanumeric code (e.g., `PAIR-A7F93E`) with a **10-minute expiration window**.
2. **Device Handshake & Token Issuance:**
   - The remote monitored device submits the pairing code along with its hardware metadata (`device_id`, `hostname`, `ip_address`, `hardware_model`, `role`) to `POST /api/pairing/exchange`.
   - The server validates the code and issues a permanent **PBKDF2-HMAC-SHA256 authentication token**.
3. **Persistent Registry & Verification:**
   - The device is committed to `paired_devices.json`.
   - Subsequent telemetry synchronization requests from this device include the token in the `Authorization: Bearer <token>` header.
4. **Instant Revocation:**
   - Administrators can immediately revoke or permanently delete any paired device from the web UI with one click.

---

## Two-Computer Distributed Deployment

For production deployments where **Computer 1** is the monitored node and **Computer 2** is the dedicated prediction & dashboard server:

```text
┌──────────────────────────────────────┐          ┌──────────────────────────────────────┐
│     COMPUTER 1: MONITORED DEVICE     │          │    COMPUTER 2: PREDICTION MACHINE    │
│            (e.g., 192.168.1.105)     │          │         (e.g., 192.168.1.110)        │
│                                      │          │                                      │
│  python overall_run.py               │          │  python overall_run.py               │
│    --role MONITORING                 │          │    --role PREDICTION                 │
│    --port 8001                       │          │    --port 8080                       │
│                                      │          │                                      │
│  * Samples physical CPU sensors      │   Sync   │  * Hosts Enterprise Control Center   │
│  * Runs Agents 1-4                   │ ───────> │  * Executes Arrhenius & Hybrid RUL   │
│  * Exposes HTTP records on port 8001 │  (LAN)   │  * Runs Ollama Agentic Diagnostics   │
└──────────────────────────────────────┘          └──────────────────────────────────────┘
```

### Manual Headless Execution (Alternative)

If you prefer executing discrete processes without `overall_run.py`:

#### Computer 1: Monitoring Machine
```bash
source .venv/bin/activate
python run.py --agents agent1 agent2 agent3 agent4 --interval 2.0 --port 8001
```

#### Computer 2: Prediction Machine
```bash
cd iqac-prediction
source ../.venv/bin/activate

# 1. Sync records continuously from Computer 1
python main.py --collect-url http://192.168.1.105:8001 --device-id target-cpu --interval 5.0

# 2. Run prognostic API and enterprise dashboard
cd ..
python overall_run.py --role PREDICTION --port 8080
```

---

## Automated Test Suite & Verification

The repository contains an exhaustive automated test suite with **55 passing tests** across 12 test modules, verifying every layer of the cyber architecture:

```bash
source .venv/bin/activate
pytest tests/ -v
```

### Comprehensive Test Coverage:
1. `tests/test_server_manager.py`: Server lifecycle, initial state, client connection registry, VPN interface discovery, and unreachable port testing.
2. `tests/test_pairing_and_auth.py`: Cryptographic PBKDF2 hashing, sliding-window rate limiting, single-use pairing code generation, code expiration, and token exchange.
3. `tests/test_enterprise_routes.py`: Server management REST endpoints, pairing workflows, hardware inventory, and time-filtered telemetry series.
4. `tests/test_rul_engine.py`: Normal CPU temperature, sustained high temperature (90°C), thermal spikes (>85°C), missing sensor handling, zero/invalid temperature rejection, missing voltage fallback, duration thresholds, and boundary conditions.
5. `tests/test_ml_and_hybrid.py`: Supervised ML regression, prognostic metrics (MAE, RMSE, MAPE, NASA PHM score), uncalibrated refusal, and hybrid Bayesian fusion.
6. `tests/test_monitoring_and_sync.py`: Recorder telemetry sanitization, out-of-order timestamps, duplicate sync prevention, network retry/timeout handling, and hardware identification.
7. `tests/test_connections.py`: Network interface detection, hardware discovery, unreachable peer ping tests, and peer pairing.
8. `tests/test_launcher.py`: Dependency verification, free port allocation, device manager initialisation, and primary IP discovery.
9. `tests/test_measurements.py`: Hardware telemetry schema validation, measurement manager lifecycle, and circular ring-buffer rotation.
10. `tests/test_services.py`: Log buffer rotation and severity filtering, process manager port detection, controlled stress tester lifecycle, and dashboard API responses.
11. `tests/test_synthetic_suite.py`: Multi-day synthetic telemetry generation, monotonicity checks, and damage rate acceleration.
12. `tests/test_api_integration.py`: FastAPI endpoints, structured RUL serialization, history generation, and dashboard rendering.

### Hardware Verification Suite (`test_run.py`):
```bash
python test_run.py
```
- **Mode 1:** 30-second physical hardware telemetry sampling loop with sensor validation and database integrity checks.
- **Mode 2:** 30-second bounded CPU stress test (75% load) with automatic 85.0°C thermal cutoff protection.

---

## Epistemic Limitations & Scientific Caveats

1. **Uncalibrated Heuristics:** Standard Arrhenius activation energy ($E_a \approx 0.7\text{ eV}$) and Coffin-Manson exponents ($q \approx 2.5$) represent accepted general physics for semiconductor devices, but actual rate constants vary across silicon fabrication nodes (e.g. 5nm FinFET vs 28nm planar). Outputs must be treated as prognostic estimates rather than certified physical truth unless calibrated against destructive lifetime benches.
2. **Sensor Precision:** Many consumer operating systems expose CPU package thermal diodes rather than true hotspot silicon junctions. Temperatures reported through `psutil` or external probes should be interpreted as lower bounds on peak junction temperatures.
# FINALISED-RUL-SYSTEM
# FINALISED-RUL-SYSTEM
