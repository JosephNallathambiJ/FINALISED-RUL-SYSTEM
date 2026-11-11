#!/usr/bin/env python3
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent4_core.config import load_config
from agent4_core.agent import CpuPerformanceAgent

def run_cli_loop(agent: CpuPerformanceAgent, steps: int = 0):
    print("\n" + "="*70)
    print(f"⚙️ AGENT4: CPU Frequency & Dynamic Scaling Agent")
    print(f"Active Profile: {agent.profile.name}")
    print("="*70 + "\n")

    step_counter = 0
    try:
        while True:
            step_counter += 1
            res = agent.step()
            freq = res["freq_data"]["freq_mhz"]
            cpu = res["telemetry"]["cpu_percent"]
            gov = res["governor"]["governor_mode"]
            u_score = res["utility"]["best_utility"]

            print(f"[Step #{res['step']:03d}] {res['profile']} | Freq: {freq} MHz | CPU: {cpu}% | Gov: {gov} (Utility: {u_score:.4f})")

            if steps > 0 and step_counter >= steps:
                break
            time.sleep(agent.config.agent_interval_seconds)
    except KeyboardInterrupt:
        print("\n[Agent4] CPU Governor loop stopped.")

def run_web_dashboard(agent: CpuPerformanceAgent, host: str = "127.0.0.1", port: int = 8004):
    import uvicorn
    from agent4_core.web.app import app, init_web_agent
    init_web_agent(agent)
    print(f"\n🚀 Launching Agent 4 Web Monitor on http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

def run_test_run(agent: CpuPerformanceAgent):
    print("Executing Agent 4 system verification...")
    for _ in range(10):
        res = agent.step()
        assert "freq_data" in res
        assert "governor" in res
    print("✅ All Agent 4 subsystems VERIFIED SUCCESSFULLY!")

def main():
    parser = argparse.ArgumentParser(description="Agent 4 - CPU Frequency AI Agent")
    parser.add_argument("--profile", choices=["laptop", "computer", "drone", "iot"], default=None)
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--steps", type=int, default=0)

    args = parser.parse_args()
    config = load_config("config.yaml")
    if args.profile:
        config.active_profile = args.profile

    agent = CpuPerformanceAgent(config=config)

    if args.test_run:
        run_test_run(agent)
    elif args.web:
        run_web_dashboard(agent, host=args.host, port=args.port)
    else:
        run_cli_loop(agent, steps=args.steps)

if __name__ == "__main__":
    main()
