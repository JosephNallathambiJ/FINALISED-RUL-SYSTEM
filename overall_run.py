#!/usr/bin/env python3
"""
Single-Click Main Entry Point & Orchestrator for CPU Monitoring & RUL Prediction System.

This script detects the host OS and environment, validates dependencies, initializes
the database, launches appropriate services based on the assigned role (MONITORING,
PREDICTION, or LOCAL_DEMO), starts the Unified Control Center dashboard, and opens
the dashboard in the user's default browser.
"""

from __future__ import annotations

import argparse
import os
import platform
import signal
import socket
import sys
import time
import webbrowser
from pathlib import Path
from typing import List, Optional

# Set working directory to project root
ROOT_DIR = Path(__file__).resolve().parent
PRED_DIR = ROOT_DIR / "iqac-prediction"
os.chdir(str(ROOT_DIR))

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

from control.device_manager import DeviceManager, get_primary_ip
from control.process_manager import ProcessManager
from control.server_manager import detect_vpn_and_tailscale, server_mgr


def print_banner(role: str, ip: str, port: int, host: str = "0.0.0.0"):
    vpn_info = detect_vpn_and_tailscale()
    vpn_line = f"  Tailscale / VPN  : http://{vpn_info['tailscale_ip']}:{port}\n" if vpn_info["tailscale_ip"] else ""
    lan_line = f"http://{ip}:{port}" if host == "0.0.0.0" else "Disabled (Localhost Only)"

    banner = f"""
================================================================================
⚡ MACHINE LEARNING FRAMEWORK FOR PHYSICAL SYSTEM FAILURE PREDICTION & RELIABILITY ASSESSMENT
================================================================================
  Host System      : {platform.system()} {platform.machine()} ({platform.release()})
  Python Runtime   : {platform.python_version()} ({sys.executable})
  Primary IPv4     : {ip}
  Operational Role : {role}
  Localhost Access : http://127.0.0.1:{port}
  Local Area (LAN) : {lan_line}
{vpn_line}================================================================================
"""
    print(banner)


def check_dependencies() -> bool:
    """Verify that all core Python libraries are installed."""
    required = [
        "fastapi",
        "uvicorn",
        "psutil",
        "pydantic",
        "jinja2",
        "websockets",
        "requests",
        "numpy",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"\n[ERROR] Missing required dependencies: {', '.join(missing)}")
        print(f"Please install them via: {sys.executable} -m pip install {' '.join(missing)}\n")
        return False
    return True


def find_free_port(preferred: int = 8080, max_tries: int = 20) -> int:
    """Return preferred port if available, else find next available port."""
    for p in range(preferred, preferred + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return preferred


def interactive_select_role(current_role: str) -> str:
    """Interactive role selection when run in an interactive terminal."""
    if not sys.stdin.isatty():
        return current_role

    print("\nSelect the Operational Role for this machine:")
    print("  [1] MONITORING COMPUTER  (Collects real CPU telemetry & syncs to remote)")
    print("  [2] PREDICTION COMPUTER  (Runs ML/Arrhenius RUL models & Agentic AI)")
    print("  [3] LOCAL DEMO (ALL-IN-ONE) (Runs full monitoring, RUL & AI on one machine)")
    print(f"  [Enter] Keep configured default [{current_role}]")

    try:
        choice = input("\nEnter choice [1-3 or Enter]: ").strip()
        if choice == "1":
            return "MONITORING"
        elif choice == "2":
            return "PREDICTION"
        elif choice == "3":
            return "LOCAL_DEMO"
        return current_role
    except (EOFError, KeyboardInterrupt):
        return current_role


def main():
    parser = argparse.ArgumentParser(description="Single-Click CPU Monitoring & RUL Prediction Control Center")
    parser.add_argument(
        "--role",
        choices=["MONITORING", "PREDICTION", "LOCAL_DEMO"],
        help="Operational role for this machine",
    )
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Network host interface (default: 0.0.0.0 for LAN access)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open web browser")
    parser.add_argument("--interactive", action="store_true", help="Prompt for role selection interactively")
    args = parser.parse_args()

    # 1. Dependency Check
    if not check_dependencies():
        sys.exit(1)

    # 2. Device Manager & Role Determination
    device_mgr = DeviceManager()
    if args.role:
        role = args.role
        device_mgr.set_role(role)
    elif args.interactive:
        role = interactive_select_role(device_mgr.role)
        device_mgr.set_role(role)
    else:
        role = device_mgr.role

    primary_ip = get_primary_ip()
    port = find_free_port(args.port)
    host = args.host

    # Register server manager parameters
    server_mgr.host = host
    server_mgr.port = port
    server_mgr.start_time = time.time()
    server_mgr.status = "RUNNING"

    print_banner(role, primary_ip, port, host=host)

    # 3. Initialize Database
    try:
        from database import init_db
        init_db()
        print("[INIT] Database initialized successfully (cpu_metrics.db).")
    except Exception as e:
        print(f"[WARN] Database initialization notice: {e}")

    # 4. Process Manager for Subprocess Supervision
    proc_mgr = ProcessManager()

    # Graceful Shutdown Handler
    def shutdown_handler(signum, frame):
        print("\n[SHUTDOWN] Signal received. Terminating all background services...")
        proc_mgr.stop_all()
        print("[SHUTDOWN] All services stopped cleanly. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 5. Launch Background Services based on Role
    python_bin = sys.executable

    if role == "MONITORING":
        print("\n[STARTUP] Starting Monitoring Computer Services...")
        # Check and start telemetry API on 8000
        if not proc_mgr.is_port_in_use(8000):
            print("  -> Starting Monitoring API on port 8000...")
            proc_mgr.start_process(
                name="MonitoringAPI",
                command=[python_bin, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
                cwd=str(PRED_DIR),
            )
        else:
            print("  -> Port 8000 is already in use; assuming existing Monitoring API instance.")

    elif role == "PREDICTION":
        print("\n[STARTUP] Starting Prediction Computer Services...")
        # Check and start prediction API on 8001
        if not proc_mgr.is_port_in_use(8001):
            print("  -> Starting Prediction API on port 8001...")
            proc_mgr.start_process(
                name="PredictionAPI",
                command=[python_bin, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"],
                cwd=str(PRED_DIR),
            )
        else:
            print("  -> Port 8001 is already in use; assuming existing Prediction API instance.")

    elif role == "LOCAL_DEMO":
        print("\n[STARTUP] Starting Local Demo Mode (Unified Monitoring + Prediction + Agent)...")
        # In Local Demo, dashboard itself hosts both the measurement supervisor and prediction routes,
        # but we also ensure API port 8001 is running for legacy API client compatibility.
        if not proc_mgr.is_port_in_use(8001):
            print("  -> Starting Local Prediction API on port 8001...")
            proc_mgr.start_process(
                name="PredictionAPI",
                command=[python_bin, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"],
                cwd=str(PRED_DIR),
            )

    # 6. Auto-open Browser
    url = f"http://127.0.0.1:{port}"
    if not args.no_browser:
        print(f"\n[DASHBOARD] Opening control center in default browser: {url}")

        def _open():
            time.sleep(1.2)
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"[WARN] Could not automatically launch browser: {e}")

        import threading
        threading.Thread(target=_open, daemon=True).start()

    # 7. Start Main Dashboard Server (Foreground)
    print(f"[SERVER] Starting Enterprise Dashboard Server on {host}:{port}...")
    import uvicorn
    try:
        uvicorn.run("dashboard.app:app", host=host, port=port, log_level="info", access_log=False)
    except Exception as e:
        print(f"[ERROR] Dashboard server error: {e}")
    finally:
        shutdown_handler(None, None)


if __name__ == "__main__":
    main()

