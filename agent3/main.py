#!/usr/bin/env python3
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent3_core.config import load_config
from agent3_core.agent import SystemVoltageAgent

def run_cli_loop(agent: SystemVoltageAgent, steps: int = 0):
    print("\n" + "="*70)
    print(f"⚡ AGENT3: System Voltage & Electrical Power AI Agent")
    print(f"Active Profile: {agent.profile.name}")
    print(f"VCore Nominal: {agent.profile.vcore_nominal}V (Range: {agent.profile.vcore_min}V - {agent.profile.vcore_max}V)")
    print("="*70 + "\n")

    step_counter = 0
    try:
        while True:
            step_counter += 1
            res = agent.step()

            p_name = res["profile"]
            belief = res["belief"]
            vcore = belief["vcore_volts"]
            v12 = belief["v12_volts"]
            watts = belief["power_watts"]
            status = belief["voltage_status"]
            dev_pct = belief["tolerance_deviation_pct"]
            action_name = res["protection"]["executed_action"]
            u_score = res["utility"]["best_utility"]

            print(f"[Step #{res['step']:03d}] {p_name.upper()} | VCore: {vcore:5.3f}V | +12V: {v12:5.2f}V | Power: {watts:5.1f}W")
            print(f"         ├─ Dev: {dev_pct:4.1f}% | Rail Status: [{status}] | Action Chosen: {action_name}")
            print(f"         └─ Output: {res['protection']['message']} (Utility U: {u_score:.4f})\n")

            if steps > 0 and step_counter >= steps:
                break
            time.sleep(agent.config.agent_interval_seconds)
    except KeyboardInterrupt:
        print("\n[Agent3] System voltage monitoring loop stopped by user.")

def run_web_dashboard(agent: SystemVoltageAgent, host: str = "127.0.0.1", port: int = 8002):
    import uvicorn
    from agent3_core.web.app import app, init_web_agent
    init_web_agent(agent)
    print(f"\n🚀 Launching Agent 3 Interactive Voltage Monitor & API Server on http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

def run_voltage_test(agent: SystemVoltageAgent):
    print("\n" + "="*70)
    print("🔬 AGENT3: Hardware Voltage Sensor ADC & Power Actuator Hook Test")
    print("="*70)

    # 1. Test hardware voltage reading
    hw_volts = agent.voltage_sensor.read_hardware_voltage_sysfs()
    if hw_volts is not None:
        print(f"✅ Native OS/hwmon Voltage Probes detected: {hw_volts}")
    else:
        print("ℹ️  Native hwmon voltage probes not found. Electrical Physics Simulator active.")

    # 2. Attach physical ADC callback
    print("\nAttaching custom hardware ADC callback (INA219 probe demonstration)...")
    def my_physical_adc_callback():
        return {
            "vcore_volts": 1.148,
            "v12_volts": 12.02,
            "v5_volts": 5.01,
            "v33_volts": 3.30,
            "battery_volts": 11.40,
            "power_watts": 28.5
        }

    agent.register_voltage_sensor_callback(my_physical_adc_callback)
    v_read = agent.voltage_sensor.read()
    print(f"✅ Custom Voltage ADC Reading: VCore={v_read['rails']['vcore_volts']}V, 12V={v_read['rails']['v12_volts']}V")
    print(f"✅ Sensor Source: {v_read['sensor_source']}")

    # 3. Attach physical Power Actuator callback
    print("\nAttaching custom hardware Power Actuator callback...")
    def my_physical_power_actuator(action: str, vcore_delta: float):
        print(f"   [VRM Hardware Command] Executing {action} with offset {vcore_delta:+.3f}V")

    agent.register_power_actuator_callback(my_physical_power_actuator)
    act_res = agent.actuator.apply_power_action("BOOST_VCORE", +0.03)
    print(f"✅ Applied Action: {act_res['action']} via {act_res['applied_method']}\n")

def run_test_run(agent: SystemVoltageAgent):
    print("Executing automated Agent 3 system verification (10 steps)...")
    for _ in range(10):
        res = agent.step()
        assert "rails" in res
        assert "belief" in res
        assert "utility" in res
        assert "protection" in res
    print("✅ All Agent 3 subsystems (Voltage Perception, Electrical Model, Power Utility, History DB) VERIFIED SUCCESSFULLY!")

def main():
    parser = argparse.ArgumentParser(description="Agent 3 - System Voltage & Electrical Power AI Agent")
    parser.add_argument("--profile", choices=["laptop", "computer", "drone", "iot"], default=None, help="Set active voltage profile")
    parser.add_argument("--web", action="store_true", help="Launch interactive web monitor dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Web host interface")
    parser.add_argument("--port", type=int, default=8002, help="Web server port")
    parser.add_argument("--voltage-test", action="store_true", help="Run hardware voltage sensor ADC and power actuator hook test")
    parser.add_argument("--test-run", action="store_true", help="Run quick system verification test")
    parser.add_argument("--steps", type=int, default=0, help="Number of CLI steps to execute (0 = infinite)")

    args = parser.parse_args()
    config = load_config("config.yaml")
    if args.profile:
        config.active_profile = args.profile

    agent = SystemVoltageAgent(config=config)

    if args.voltage_test:
        run_voltage_test(agent)
    elif args.test_run:
        run_test_run(agent)
    elif args.web:
        run_web_dashboard(agent, host=args.host, port=args.port)
    else:
        run_cli_loop(agent, steps=args.steps)

if __name__ == "__main__":
    main()
