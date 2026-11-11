#!/usr/bin/env python3
"""Production entry point for the separate RUL prediction system."""

from __future__ import annotations

import argparse
import threading

import uvicorn

from api import app
from database import init_db
from monitor_client import collect_daily_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Start IQAC telemetry ingestion and RUL prediction API")
    parser.add_argument(
        "--monitor-url",
        help="Target monitoring API URL, for example http://192.168.1.20:8001",
    )
    parser.add_argument("--device-id", default="target-device")
    parser.add_argument("--collect-interval", type=float, default=5.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.collect_interval <= 0:
        parser.error("--collect-interval must be greater than zero")

    init_db()
    if args.monitor_url:
        collector = threading.Thread(
            target=collect_daily_records,
            args=(args.monitor_url, args.device_id, args.collect_interval),
            daemon=True,
            name="telemetry-collector",
        )
        collector.start()
        print(f"Collecting telemetry from {args.monitor_url} for {args.device_id}")
    else:
        print("No --monitor-url supplied; API started for POST /telemetry clients")

    print(f"RUL prediction API listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
