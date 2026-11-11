"""Poll a monitoring agent on the target machine and store validated telemetry."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional, Set
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from database import init_db, insert_daily_record, insert_telemetry, update_sync_state


def _find_value(data: Any, names: Set[str], default: Any = None) -> Any:
    if isinstance(data, dict):
        for name in names:
            if name in data and data[name] is not None:
                return data[name]
        for value in data.values():
            found = _find_value(value, names)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = _find_value(value, names)
            if found is not None:
                return found
    return default


def normalize_status(status: Dict[str, Any], device_id: str) -> Dict[str, Any]:
    """Validate and normalize raw incoming telemetry from a monitoring agent."""
    raw_temp = _find_value(
        status,
        {"cpu_temp", "temperature_c", "temperature_celsius", "hw_temperature_c"},
        None,
    )
    valid_temp: Optional[float] = None
    if raw_temp is not None:
        try:
            t = float(raw_temp)
            if 5.0 <= t <= 125.0:
                valid_temp = round(t, 2)
        except (ValueError, TypeError):
            valid_temp = None

    fan_rpm = int(float(_find_value(status, {"fan_rpm", "current_rpm", "rpm"}, 0) or 0))
    cpu_usage = float(_find_value(status, {"cpu_usage", "cpu_percent"}, 0.0) or 0.0)
    cpu_usage = max(0.0, min(100.0, cpu_usage))

    voltage_v = _find_value(status, {"voltage_v", "vcore_volts", "vcore"}, None)
    valid_voltage: Optional[float] = None
    if voltage_v is not None:
        try:
            v = float(voltage_v)
            if 0.3 <= v <= 3.5:
                valid_voltage = round(v, 3)
        except (ValueError, TypeError):
            pass

    freq_mhz = _find_value(status, {"cpu_freq_mhz", "freq_mhz"}, None)
    valid_freq: Optional[float] = None
    if freq_mhz is not None:
        try:
            f = float(freq_mhz)
            if 0.0 <= f <= 10000.0:
                valid_freq = round(f, 1)
        except (ValueError, TypeError):
            pass

    source_spike = _find_value(status, {"is_spike", "spike"}, False)
    if isinstance(source_spike, str):
        source_spike = source_spike.strip().lower() in {"1", "true", "yes", "on"}
    is_spike = bool(source_spike) or (valid_temp is not None and valid_temp >= 85.0)

    throttled = _find_value(status, {"is_throttled", "throttling"}, False)
    if isinstance(throttled, str):
        throttled = throttled.strip().lower() in {"1", "true", "yes", "on"}

    source_timestamp = _find_value(status, {"timestamp", "recorded_at"})
    if isinstance(source_timestamp, (int, float)):
        source_timestamp = datetime.datetime.fromtimestamp(
            source_timestamp, datetime.timezone.utc
        ).isoformat()
    elif not source_timestamp:
        source_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        "device_id": device_id,
        "cpu_temp": valid_temp,
        "fan_rpm": fan_rpm,
        "cpu_usage": cpu_usage,
        "voltage_v": valid_voltage,
        "cpu_freq_mhz": valid_freq,
        "is_spike": is_spike,
        "is_throttled": bool(throttled),
        "timestamp": source_timestamp,
    }


def fetch_with_retry(
    url: str,
    timeout: float = 10.0,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    backoff: float = 1.5,
) -> Dict[str, Any]:
    """Execute HTTP GET with exponential backoff and authentication headers."""
    headers = {"Accept": "application/json"}
    key = api_key or os.getenv("IQAC_API_KEY")
    if key:
        headers["X-API-Key"] = key

    last_error: Optional[Exception] = None
    delay = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except (URLError, TimeoutError, ConnectionResetError) as err:
            last_error = err
            if attempt < max_retries:
                time.sleep(delay)
                delay *= backoff
        except HTTPError as http_err:
            if http_err.code in (502, 503, 504) and attempt < max_retries:
                time.sleep(delay)
                delay *= backoff
            else:
                raise http_err

    raise ConnectionError(f"Failed to connect to {url} after {max_retries} attempts: {last_error}")


def collect(
    monitor_url: str,
    device_id: str,
    interval: float,
    api_key: Optional[str] = None,
) -> None:
    """Continuously poll live status from the monitoring machine."""
    init_db()
    endpoint = monitor_url.rstrip("/") + "/api/status"
    print(f"Collecting live telemetry from {endpoint} for device '{device_id}' (interval: {interval}s)")
    while True:
        try:
            status = fetch_with_retry(endpoint, timeout=min(max(interval, 2.0), 30.0), api_key=api_key)
            normalized = normalize_status(status, device_id)
            insert_telemetry(**normalized)
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Saved telemetry: {normalized}")
        except Exception as error:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Monitoring source unavailable: {error}")
        time.sleep(interval)


def sync_daily_records(
    monitor_url: str,
    device_id: str,
    timeout: float = 10.0,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Fetch all compact records from the monitoring machine and upsert them safely."""
    init_db()
    endpoint = monitor_url.rstrip("/") + "/records"
    payload = fetch_with_retry(endpoint, timeout=timeout, api_key=api_key, max_retries=max_retries)

    daily = payload.get("daily", [])
    spikes = payload.get("spikes", [])
    hardware_info = payload.get("hardware_info", {})
    duration_hours = float(payload.get("monitoring_duration_hours", 0.0) or 0.0)

    # Upsert each daily record
    for record in daily:
        insert_daily_record(device_id, record)

    total_readings = sum(int(record.get("reading_count", 0)) for record in daily)
    result = {
        "source": endpoint,
        "daily_records": len(daily),
        "spike_events": len(spikes),
        "total_readings": total_readings,
        "monitoring_duration_hours": duration_hours,
        "hardware_info": hardware_info,
        "synchronized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    update_sync_state(
        device_id=device_id,
        duration_hours=duration_hours,
        daily_records=len(daily),
        total_readings=total_readings,
        spike_events=len(spikes),
        hardware_info=hardware_info,
    )
    return result


def collect_daily_records(
    monitor_url: str,
    device_id: str,
    interval: float,
    api_key: Optional[str] = None,
) -> None:
    """Continuously synchronize compact records over the network."""
    init_db()
    endpoint = monitor_url.rstrip("/") + "/records"
    print(f"Synchronizing records from {endpoint} for device '{device_id}' every {interval}s")
    while True:
        try:
            result = sync_daily_records(
                monitor_url, device_id, timeout=min(max(interval, 2.0), 30.0), api_key=api_key
            )
            print(
                f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Synchronized: "
                f"{result['daily_records']} days, {result['spike_events']} spikes, "
                f"{result['total_readings']} readings, {result['monitoring_duration_hours']:.2f} hrs"
            )
        except Exception as error:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Monitoring records sync failed: {error}")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hardware telemetry from a monitoring machine")
    parser.add_argument("--monitor-url", required=True, help="Target machine URL, e.g. http://192.168.1.20:8001")
    parser.add_argument("--device-id", required=True, help="Stable identifier for the monitored machine")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")
    parser.add_argument("--api-key", default=None, help="Optional API authentication key")
    args = parser.parse_args()
    collect(args.monitor_url, args.device_id, args.interval, api_key=args.api_key)
