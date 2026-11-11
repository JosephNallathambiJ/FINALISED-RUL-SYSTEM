#!/usr/bin/env python3
"""Agent5 CLI entry point.
Provides subcommands to start, stop, and query status of the supervisor.
"""
import argparse
import signal
import sys
import threading
from pathlib import Path

from agent5_core.config import load_config
from agent5_core.supervisor import Supervisor


def run_supervisor(config_path: str):
    cfg = load_config(config_path)
    supervisor = Supervisor(cfg)
    # Handle graceful shutdown
    def shutdown(sig, frame):
        print("Received signal, shutting down supervisor...")
        supervisor.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    supervisor.start()
    # Keep main thread alive while supervisor runs in background threads
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        shutdown(None, None)


def main():
    parser = argparse.ArgumentParser(prog="agent5", description="Agent5 Supervisor CLI")
    parser.add_argument("--config", default="config.yaml", help="Path to supervisor config file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="Start the supervisor and manage agents")
    subparsers.add_parser("status", help="Print a quick JSON status snapshot")
    subparsers.add_parser("stop", help="Stop all agents and exit (requires running process PID) – placeholder")

    args = parser.parse_args()

    if args.command == "run":
        run_supervisor(args.config)
    elif args.command == "status":
        cfg = load_config(args.config)
        supervisor = Supervisor(cfg)
        # Start supervisor briefly to gather status then stop
        supervisor.start()
        # Allow a short moment for processes to start
        import time
        time.sleep(1)
        overview = supervisor.get_overview()
        import json
        print(json.dumps(overview, indent=2))
        supervisor.stop()
    elif args.command == "stop":
        # In a real deployment this would signal the running process via PID file or similar.
        print("Stop command not implemented in this stub.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
