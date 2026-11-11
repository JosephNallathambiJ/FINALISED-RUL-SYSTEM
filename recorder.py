#!/usr/bin/env python3
"""Compact daily telemetry, hardware identification, and spike recorder for the monitored device."""

from __future__ import annotations

from contextlib import asynccontextmanager
import datetime as dt
import json
import os
import platform
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

DB_PATH = Path(__file__).with_name("monitoring_records.db")
DB_LOCK = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="IQAC Monitoring Records", lifespan=lifespan)

# Optional API authentication key
API_KEY: Optional[str] = os.getenv("IQAC_API_KEY")


def get_hardware_identity() -> Dict[str, Any]:
    """Inspect and return local CPU and platform identification."""
    info: Dict[str, Any] = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "Unknown CPU",
        "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
    }
    try:
        freq = psutil.cpu_freq()
        if freq:
            info["cpu_freq_max_mhz"] = freq.max
            info["cpu_freq_min_mhz"] = freq.min
    except Exception:
        pass
    return info


def init_db() -> None:
    """Initialize database and safely apply non-destructive schema migrations."""
    with DB_LOCK, sqlite3.connect(DB_PATH) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS daily_metrics (
                day TEXT PRIMARY KEY,
                reading_count INTEGER NOT NULL,
                temperature_sum REAL NOT NULL,
                temperature_max REAL NOT NULL,
                temperature_min REAL NOT NULL,
                valid_temp_count INTEGER NOT NULL DEFAULT 0,
                cpu_sum REAL NOT NULL,
                rpm_sum REAL NOT NULL,
                voltage_sum REAL NOT NULL DEFAULT 0.0,
                valid_voltage_count INTEGER NOT NULL DEFAULT 0,
                freq_sum REAL NOT NULL DEFAULT 0.0,
                valid_freq_count INTEGER NOT NULL DEFAULT 0,
                throttling_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS spike_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                agent TEXT NOT NULL,
                temperature_c REAL NOT NULL,
                cpu_usage REAL NOT NULL,
                fan_rpm REAL NOT NULL,
                details TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recorder_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_spikes_time ON spike_events(recorded_at);
            """
        )
        # Apply non-destructive column migrations for legacy DBs
        cursor = connection.cursor()
        cursor.execute("PRAGMA table_info(daily_metrics)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        new_columns = [
            ("valid_temp_count", "INTEGER NOT NULL DEFAULT 0"),
            ("voltage_sum", "REAL NOT NULL DEFAULT 0.0"),
            ("valid_voltage_count", "INTEGER NOT NULL DEFAULT 0"),
            ("freq_sum", "REAL NOT NULL DEFAULT 0.0"),
            ("valid_freq_count", "INTEGER NOT NULL DEFAULT 0"),
            ("throttling_count", "INTEGER NOT NULL DEFAULT 0"),
        ]
        for col_name, col_type in new_columns:
            if col_name not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE daily_metrics ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass

        # Save hardware identity to state
        hw_json = json.dumps(get_hardware_identity())
        cursor.execute(
            "INSERT INTO recorder_state (key, value) VALUES ('hardware_identity', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (hw_json,),
        )
        connection.commit()


def record_reading(
    temperature_c: Optional[float] = None,
    cpu_usage: float = 0.0,
    fan_rpm: float = 0.0,
    voltage_v: Optional[float] = None,
    cpu_freq_mhz: Optional[float] = None,
    throttling: bool = False,
    agent: str = "monitor",
    recorded_at: Optional[str] = None,
    is_spike: bool = False,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate and record an individual or combined sensor observation."""
    recorded_at = recorded_at or dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        timestamp = dt.datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
    except ValueError:
        timestamp = dt.datetime.now(dt.timezone.utc)
        recorded_at = timestamp.isoformat()
    day = timestamp.date().isoformat()

    # Telemetry input validation and sanity checking
    cpu_usage = float(max(0.0, min(100.0, cpu_usage or 0.0)))
    fan_rpm = float(max(0.0, min(25000.0, fan_rpm or 0.0)))

    # Temperature validation: Reject 0, negative, or absurd values (>125°C)
    valid_temp = False
    temp_val = 0.0
    if temperature_c is not None:
        try:
            t = float(temperature_c)
            if 5.0 <= t <= 125.0:
                valid_temp = True
                temp_val = t
        except (ValueError, TypeError):
            valid_temp = False

    # Voltage validation
    valid_voltage = False
    volt_val = 0.0
    if voltage_v is not None:
        try:
            v = float(voltage_v)
            if 0.3 <= v <= 3.5:
                valid_voltage = True
                volt_val = v
        except (ValueError, TypeError):
            valid_voltage = False

    # Frequency validation
    valid_freq = False
    freq_val = 0.0
    if cpu_freq_mhz is not None:
        try:
            f = float(cpu_freq_mhz)
            if 0.5 <= f < 100.0:
                f *= 1000.0  # Scale GHz to MHz
            if 100.0 <= f <= 10000.0:
                valid_freq = True
                freq_val = f
        except (ValueError, TypeError):
            valid_freq = False

    with DB_LOCK, sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()

        # Update first_seen and last_seen timestamps
        cursor.execute(
            "INSERT OR IGNORE INTO recorder_state (key, value) VALUES ('first_seen_at', ?)",
            (recorded_at,),
        )

        # Calculate accumulated active operating hours (prevents sleep/reboot inflation)
        cursor.execute("SELECT value FROM recorder_state WHERE key = 'last_tick_epoch'")
        last_tick_row = cursor.fetchone()
        current_epoch = timestamp.timestamp()
        active_delta_sec = 0.0
        if last_tick_row:
            try:
                prev_epoch = float(last_tick_row[0])
                delta = current_epoch - prev_epoch
                # If delta is between 0.1s and 120s, system was actively running
                if 0.1 <= delta <= 120.0:
                    active_delta_sec = delta
            except ValueError:
                pass

        cursor.execute(
            "INSERT INTO recorder_state (key, value) VALUES ('last_tick_epoch', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(current_epoch),),
        )

        cursor.execute("SELECT value FROM recorder_state WHERE key = 'accumulated_active_seconds'")
        accum_row = cursor.fetchone()
        curr_accum = float(accum_row[0]) if accum_row else 0.0
        new_accum = curr_accum + active_delta_sec
        cursor.execute(
            "INSERT INTO recorder_state (key, value) VALUES ('accumulated_active_seconds', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(new_accum),),
        )

        cursor.execute(
            "INSERT INTO recorder_state (key, value) VALUES ('last_seen_at', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (recorded_at,),
        )

        # Check existing row for today
        cursor.execute(
            "SELECT reading_count, temperature_sum, temperature_max, temperature_min, valid_temp_count, "
            "cpu_sum, rpm_sum, voltage_sum, valid_voltage_count, freq_sum, valid_freq_count, throttling_count "
            "FROM daily_metrics WHERE day = ?",
            (day,),
        )
        existing = cursor.fetchone()

        if existing:
            (
                r_cnt, t_sum, t_max, t_min, v_t_cnt,
                c_sum, m_sum, v_sum, v_v_cnt, f_sum, v_f_cnt, thr_cnt
            ) = existing

            r_cnt += 1
            c_sum += cpu_usage
            m_sum += fan_rpm
            if valid_temp:
                t_sum += temp_val
                v_t_cnt += 1
                t_max = max(t_max, temp_val)
                t_min = min(t_min, temp_val) if t_min > 0 else temp_val
            if valid_voltage:
                v_sum += volt_val
                v_v_cnt += 1
            if valid_freq:
                f_sum += freq_val
                v_f_cnt += 1
            if throttling:
                thr_cnt += 1

            cursor.execute(
                """UPDATE daily_metrics SET
                    reading_count = ?, temperature_sum = ?, temperature_max = ?, temperature_min = ?,
                    valid_temp_count = ?, cpu_sum = ?, rpm_sum = ?, voltage_sum = ?,
                    valid_voltage_count = ?, freq_sum = ?, valid_freq_count = ?, throttling_count = ?,
                    updated_at = ?
                WHERE day = ?""",
                (
                    r_cnt, t_sum, t_max, t_min, v_t_cnt, c_sum, m_sum,
                    v_sum, v_v_cnt, f_sum, v_f_cnt, thr_cnt, recorded_at, day
                ),
            )
        else:
            init_t_max = temp_val if valid_temp else 0.0
            init_t_min = temp_val if valid_temp else 0.0
            init_v_t_cnt = 1 if valid_temp else 0
            init_v_v_cnt = 1 if valid_voltage else 0
            init_v_f_cnt = 1 if valid_freq else 0
            init_thr_cnt = 1 if throttling else 0

            cursor.execute(
                """INSERT INTO daily_metrics
                (day, reading_count, temperature_sum, temperature_max, temperature_min, valid_temp_count,
                 cpu_sum, rpm_sum, voltage_sum, valid_voltage_count, freq_sum, valid_freq_count,
                 throttling_count, updated_at)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    day, temp_val, init_t_max, init_t_min, init_v_t_cnt,
                    cpu_usage, fan_rpm, volt_val, init_v_v_cnt, freq_val, init_v_f_cnt,
                    init_thr_cnt, recorded_at
                ),
            )

        if is_spike:
            # Prevent recording identical spikes in the same second
            cursor.execute(
                "SELECT COUNT(*) FROM spike_events WHERE recorded_at = ? AND agent = ?",
                (recorded_at, agent),
            )
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    """INSERT INTO spike_events
                    (recorded_at, agent, temperature_c, cpu_usage, fan_rpm, details)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (recorded_at, agent, temp_val, cpu_usage, fan_rpm, json.dumps(details or {})),
                )

        connection.commit()


def get_records(since: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve all synchronized daily summaries, spike events, and verified hardware state."""
    with DB_LOCK, sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        daily_query = "SELECT * FROM daily_metrics"
        params: tuple[Any, ...] = ()
        if since:
            daily_query += " WHERE day >= ?"
            params = (since,)
        daily_query += " ORDER BY day"
        daily = [dict(row) for row in connection.execute(daily_query, params)]
        spikes = [dict(row) for row in connection.execute("SELECT * FROM spike_events ORDER BY id")]
        state = dict(connection.execute("SELECT key, value FROM recorder_state").fetchall())

    for row in daily:
        v_t_cnt = row.get("valid_temp_count", 0)
        t_sum = row.pop("temperature_sum", 0.0)
        row["average_temperature_c"] = (t_sum / v_t_cnt) if v_t_cnt > 0 else None

        r_cnt = max(1, row["reading_count"])
        row["average_cpu_usage"] = round(row.pop("cpu_sum", 0.0) / r_cnt, 2)
        row["average_fan_rpm"] = round(row.pop("rpm_sum", 0.0) / r_cnt, 1)

        v_v_cnt = row.get("valid_voltage_count", 0)
        v_sum = row.pop("voltage_sum", 0.0)
        row["average_voltage_v"] = round(v_sum / v_v_cnt, 3) if v_v_cnt > 0 else None

        v_f_cnt = row.get("valid_freq_count", 0)
        f_sum = row.pop("freq_sum", 0.0)
        row["average_cpu_freq_mhz"] = round(f_sum / v_f_cnt, 1) if v_f_cnt > 0 else None

        row["spike_count"] = sum(1 for spike in spikes if spike["recorded_at"][:10] == row["day"])

    # Duration calculation using accumulated active seconds, with fallback to timestamps
    duration_hours = 0.0
    if "accumulated_active_seconds" in state:
        try:
            duration_hours = float(state["accumulated_active_seconds"]) / 3600.0
        except ValueError:
            pass

    if duration_hours <= 0.0:
        try:
            duration_hours = max(
                0.0,
                (
                    dt.datetime.fromisoformat(state["last_seen_at"].replace("Z", "+00:00")).timestamp()
                    - dt.datetime.fromisoformat(state["first_seen_at"].replace("Z", "+00:00")).timestamp()
                )
                / 3600.0,
            )
        except (KeyError, TypeError, ValueError):
            pass

    hardware_info = {}
    if "hardware_identity" in state:
        try:
            hardware_info = json.loads(state["hardware_identity"])
        except json.JSONDecodeError:
            pass

    return {
        "daily": daily,
        "spikes": spikes,
        "monitoring_duration_hours": round(duration_hours, 3),
        "hardware_info": hardware_info,
        "first_seen_at": state.get("first_seen_at"),
        "last_seen_at": state.get("last_seen_at"),
    }


def verify_auth(x_api_key: Optional[str] = None, authorization: Optional[str] = None) -> None:
    """Verify API authorization when IQAC_API_KEY is configured."""
    if not API_KEY:
        return
    token = x_api_key
    if not token and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")





@app.get("/")
def health() -> Dict[str, Any]:
    return {
        "status": "monitoring",
        "service": "iqac-recording-agent",
        "hardware": get_hardware_identity(),
    }


@app.get("/records")
def records(
    since: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    verify_auth(x_api_key, authorization)
    return get_records(since)


init_db()
