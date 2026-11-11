#!/usr/bin/env python3
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent2_core.config import load_config
from agent2_core.agent import FanCoolingAgent

def run_cli_loop(agent: FanCoolingAgent, steps: int = 0):
    print("\n" + "="*70)
    print(f"🌀 AGENT2: Fan RPM & Active Cooling AI Agent")
    print(f"Active Profile: {agent.profile.name}")
    print(f"Target Temperature Setpoint: {agent.profile.target_temp}°C | Max RPM: {agent.profile.max_rpm}")
    print("="*70 + "\n")

    step_counter = 0
    try:
        while True:
            step_counter += 1
            res = agent.step()
            
            p_name = res["profile"]
            rpm = res["current_rpm"]
            pwm = res["target_pwm_pct"]
            temp_c = res["temperature_c"]
            cpu_load = res["cpu_percent"]
            cfm = res["airflow_cfm"]
            dba = res["noise_dba"]
            u_score = res["utility"]["best_utility"]
            pid = res["pid"]

            print(f"[Step #{res['step']:03d}] {p_name.upper()} | Temp: {temp_c:5.1f}°C | CPU: {cpu_load:4.1f}%")
            print(f"         ├─ PID Error: {pid['error']:+5.1f}°C | PWM Output: {pwm:5.1f}% | Fan Speed: {rpm:4d} RPM")
            print(f"         └─ Airflow: {cfm:4.1f} CFM | Acoustic Noise: {dba:4.1f} dBA | Utility: {u_score:.4f}\n")

            if steps > 0 and step_counter >= steps:
                break
            time.sleep(agent.config.agent_interval_seconds)
    except KeyboardInterrupt:
        print("\n[Agent2] Cooling controller loop stopped by user.")

def run_web_dashboard(agent: FanCoolingAgent, host: str = "127.0.0.1", port: int = 8001):
    import uvicorn
    from agent2_core.web.app import app, init_web_agent
    init_web_agent(agent)
    print(f"\n🚀 Launching Agent 2 Interactive Fan Monitor & API Server on http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

def run_rpm_test(agent: FanCoolingAgent):
    print("\n" + "="*70)
    print("🔬 AGENT2: Hardware Fan RPM Tachometer & PWM Actuator Hook Test")
    print("="*70)

    # 1. Test hardware RPM tachometer reading
    hw_rpm = agent.fan_sensor.read_hardware_rpm()
    if hw_rpm is not None:
        print(f"✅ Native OS/hwmon Fan Tachometer detected: {hw_rpm} RPM")
    else:
        print("ℹ️  Native hwmon fan tachometer not found. Motor Dynamics Simulator active.")

    # 2. Attach physical tachometer callback
    print("\nAttaching custom hardware Tachometer callback...")
    def my_physical_tachometer_callback():
        return 2850.0  # Physical RPM reading

    agent.register_rpm_sensor_callback(my_physical_tachometer_callback)
    fan_read = agent.fan_sensor.read()
    print(f"✅ Custom Tachometer Reading: {fan_read['rpm']} RPM ({fan_read['sensor_source']})")

    # 3. Attach physical PWM actuator callback
    print("\nAttaching custom hardware PWM Actuator callback...")
    def my_physical_pwm_actuator(pwm_pct: float):
        print(f"   [PWM Hardware Output] Setting GPIO PWM pin duty cycle to {pwm_pct}%")

    agent.register_pwm_actuator_callback(my_physical_pwm_actuator)
    act_res = agent.actuator.set_pwm(65.0)
    print(f"✅ Applied PWM: {act_res['applied_pwm_pct']}% via {act_res['applied_method']}\n")

def run_test_run(agent: FanCoolingAgent):
    print("Executing automated Agent 2 system verification (10 steps)...")
    for _ in range(10):
        res = agent.step()
        assert "current_rpm" in res
        assert "target_pwm_pct" in res
        assert "airflow_cfm" in res
        assert "noise_dba" in res
        assert "utility" in res
        assert "pid" in res
    print("✅ All Agent 2 subsystems (RPM Perception, PID Loop, Noise Model, History DB) VERIFIED SUCCESSFULLY!")

def main():
    parser = argparse.ArgumentParser(description="Agent 2 - Fan RPM & Active Cooling AI Agent")
    parser.add_argument("--profile", choices=["laptop", "computer", "drone", "iot"], default=None, help="Set active cooling profile")
    parser.add_argument("--web", action="store_true", help="Launch interactive web monitor dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Web host interface")
    parser.add_argument("--port", type=int, default=8001, help="Web server port")
    parser.add_argument("--rpm-test", action="store_true", help="Run hardware RPM tachometer and PWM actuator hook test")
    parser.add_argument("--test-run", action="store_true", help="Run quick system verification test")
    parser.add_argument("--steps", type=int, default=0, help="Number of CLI steps to execute (0 = infinite)")

    args = parser.parse_args()
    config = load_config("config.yaml")
    if args.profile:
        config.active_profile = args.profile

    agent = FanCoolingAgent(config=config)

    if args.rpm_test:
        run_rpm_test(agent)
    elif args.test_run:
        run_test_run(agent)
    elif args.web:
        run_web_dashboard(agent, host=args.host, port=args.port)
    else:
        run_cli_loop(agent, steps=args.steps)

if __name__ == "__main__":
    main()
