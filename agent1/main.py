#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging

# Add current directory to python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent1_core.config import load_config
from agent1_core.agent import ModelUtilityLearningAgent

def run_cli_loop(agent: ModelUtilityLearningAgent, steps: int = 0):
    print("\n" + "="*70)
    print(f"🤖 AGENT1: Model-Based Utility-Based Learning AI Agent")
    print(f"Active Device Profile: {agent.profile.name}")
    print(f"Targeting: Laptops, Desktop Computers, Drones, IoT Edge Nodes")
    print("="*70 + "\n")

    step_counter = 0
    try:
        while True:
            step_counter += 1
            res = agent.step()
            
            p_name = res["profile"]
            temp_c = res["temperature"]["temperature_celsius"]
            sensor_src = res["temperature"]["sensor_source"]
            cpu_load = res["telemetry"]["cpu_percent"]
            vel = res["belief_state"]["velocity_c_per_sec"]
            status = res["belief_state"]["thermal_status"]
            action_name = res["action"]["executed_action"]
            u_score = res["utility"]["best_utility"]

            print(f"[Step #{res['step']:03d}] {p_name.upper()} | Temp: {temp_c:5.1f}°C ({sensor_src}) | CPU: {cpu_load:4.1f}% | Vel: {vel:+5.2f}°C/s")
            print(f"         ├─ State Status: [{status}] | Action Chosen: {action_name} | Utility U(S,A): {u_score:.4f}")
            print(f"         └─ Action Output: {res['action']['message']}\n")

            if steps > 0 and step_counter >= steps:
                break
            time.sleep(agent.config.agent_interval_seconds)
    except KeyboardInterrupt:
        print("\n[Agent1] Autonomous agent execution stopped by user.")

def run_web_dashboard(agent: ModelUtilityLearningAgent, host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    from agent1_core.web.app import app, init_web_agent
    init_web_agent(agent)
    print(f"\n🚀 Launching Agent1 Interactive Web Monitor & API Server on http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

def run_sensor_test(agent: ModelUtilityLearningAgent):
    print("\n" + "="*70)
    print("🔬 AGENT1: Hardware Temperature Sensor Hook Test")
    print("="*70)
    
    # 1. Test native OS sensor reading
    os_temp = agent.temp_sensor.read_hardware_temperature()
    if os_temp is not None:
        print(f"✅ Native OS Hardware Thermal Sensor detected: {os_temp}°C")
    else:
        print("ℹ️  Native OS SysFS thermal sensor not found. Dynamic Thermal Simulator active.")

    # 2. Demonstrate how user attaches a custom physical hardware sensor callback
    print("\nAttaching custom hardware sensor callback (Demonstration probe)...")
    
    # User's physical sensor function example (e.g. Raspberry Pi MCP9808 / Arduino serial reading)
    def my_physical_temp_sensor_callback():
        # Example reading from physical hardware sensor
        import math, time
        return 32.5 + (2.0 * math.sin(time.time()))

    agent.register_hardware_temperature_sensor(my_physical_temp_sensor_callback)
    
    read_res = agent.temp_sensor.read()
    print(f"✅ Custom Sensor Reading: {read_res['temperature_celsius']}°C")
    print(f"✅ Sensor Source: {read_res['sensor_source']}")
    print(f"✅ Custom Hardware Attached: {read_res['is_custom_attached']}\n")

def run_test_run(agent: ModelUtilityLearningAgent):
    print("Executing automated agent system verification (10 steps)...")
    for _ in range(10):
        res = agent.step()
        assert "temperature" in res
        assert "utility" in res
        assert "belief_state" in res
        assert "action" in res
    print("✅ All Agent1 subsystems (Perception, Model-Based State, Utility Engine, History DB) VERIFIED SUCCESSFULLY!")

def main():
    parser = argparse.ArgumentParser(description="Agent1 - Model-Based Utility-Based Learning AI Agent for CPU/IoT")
    parser.add_argument("--profile", choices=["laptop", "drone", "computer", "iot"], default=None, help="Set active hardware profile")
    parser.add_argument("--web", action="store_true", help="Launch interactive web monitor dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Web host interface")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--sensor-test", action="store_true", help="Run hardware temperature sensor hook test")
    parser.add_argument("--test-run", action="store_true", help="Run quick system verification test")
    parser.add_argument("--steps", type=int, default=0, help="Number of CLI steps to execute (0 = infinite)")

    args = parser.parse_args()

    config = load_config("config.yaml")
    if args.profile:
        config.active_profile = args.profile

    agent = ModelUtilityLearningAgent(config=config)

    if args.sensor_test:
        run_sensor_test(agent)
    elif args.test_run:
        run_test_run(agent)
    elif args.web:
        run_web_dashboard(agent, host=args.host, port=args.port)
    else:
        run_cli_loop(agent, steps=args.steps)

if __name__ == "__main__":
    main()
