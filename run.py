#!/usr/bin/env python3
"""Production entry point for continuous device monitoring."""

from __future__ import annotations

import argparse
import threading
import uvicorn

from simulation import monitor_loop
from recorder import app as recorder_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Start continuous IQAC device monitoring")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["agent1", "agent2", "agent3", "agent4"],
        help="Monitoring agents to run",
    )
    parser.add_argument("--steps", type=int, default=0, help="Number of monitor cycles (0 for continuous)")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between sensor cycles")
    parser.add_argument("--host", default="0.0.0.0", help="Network interface for the records API")
    parser.add_argument("--port", type=int, default=8001, help="Port for the records API")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    api_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": recorder_app, "host": args.host, "port": args.port, "log_level": "warning"},
        daemon=True,
    )
    api_thread.start()
    print(f"Monitoring records API: http://{args.host}:{args.port}/records")
    return monitor_loop(args, record_readings=True, publish_shared_file=False)


if __name__ == "__main__":
    raise SystemExit(main())
