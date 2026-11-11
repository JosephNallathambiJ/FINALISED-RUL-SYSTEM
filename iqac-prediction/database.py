"""SQLite persistence and metric aggregation for the prediction machine."""

from __future__ import annotations

import datetime
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
DB_FILE = str(PROJECT_DIR / "telemetry.db")


def get_db_connection() -> sqlite3.Connection:
    """Return an active connection to the SQLite metrics database."""
    return sqlite3.connect(DB_FILE)



def init_db() -> None:
    """Create database tables and apply non-destructive schema migrations."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                cpu_temp REAL,
                fan_rpm INTEGER,
                cpu_usage REAL,
                voltage_v REAL DEFAULT 1.2,
                cpu_freq_mhz REAL DEFAULT 0.0,
                is_spike BOOLEAN DEFAULT 0,
                is_throttled BOOLEAN DEFAULT 0,
                timestamp DATETIME NOT NULL
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_metrics (
                device_id TEXT NOT NULL,
                day TEXT NOT NULL,
                reading_count INTEGER NOT NULL,
                average_temperature_c REAL,
                average_fan_rpm REAL NOT NULL DEFAULT 0.0,
                average_cpu_usage REAL NOT NULL DEFAULT 0.0,
                average_voltage_v REAL DEFAULT 1.2,
                average_cpu_freq_mhz REAL DEFAULT 0.0,
                spike_count INTEGER NOT NULL DEFAULT 0,
                throttling_count INTEGER NOT NULL DEFAULT 0,
                data_quality TEXT DEFAULT 'ACCEPTABLE',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (device_id, day)
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
                device_id TEXT PRIMARY KEY,
                monitoring_duration_hours REAL NOT NULL DEFAULT 0,
                daily_records INTEGER NOT NULL DEFAULT 0,
                total_readings INTEGER NOT NULL DEFAULT 0,
                spike_events INTEGER NOT NULL DEFAULT 0,
                hardware_info TEXT DEFAULT '{}',
                synchronized_at TEXT NOT NULL
            );
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_dev_time ON telemetry(device_id, timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_dev_day ON daily_metrics(device_id, day);")

        # Migrate telemetry table columns if needed
        cursor.execute("PRAGMA table_info(telemetry)")
        tel_cols = {row[1] for row in cursor.fetchall()}
        for col_name, col_def in [
            ("voltage_v", "REAL DEFAULT 1.2"),
            ("cpu_freq_mhz", "REAL DEFAULT 0.0"),
            ("is_throttled", "BOOLEAN DEFAULT 0"),
        ]:
            if col_name not in tel_cols:
                try:
                    cursor.execute(f"ALTER TABLE telemetry ADD COLUMN {col_name} {col_def}")
                except Exception:
                    pass

        # Migrate daily_metrics table columns if needed
        cursor.execute("PRAGMA table_info(daily_metrics)")
        daily_cols = {row[1] for row in cursor.fetchall()}
        for col_name, col_def in [
            ("average_voltage_v", "REAL DEFAULT 1.2"),
            ("average_cpu_freq_mhz", "REAL DEFAULT 0.0"),
            ("throttling_count", "INTEGER NOT NULL DEFAULT 0"),
            ("data_quality", "TEXT DEFAULT 'ACCEPTABLE'"),
        ]:
            if col_name not in daily_cols:
                try:
                    cursor.execute(f"ALTER TABLE daily_metrics ADD COLUMN {col_name} {col_def}")
                except Exception:
                    pass

        cursor.execute("PRAGMA table_info(sync_state)")
        sync_cols = {row[1] for row in cursor.fetchall()}
        if "hardware_info" not in sync_cols:
            try:
                cursor.execute("ALTER TABLE sync_state ADD COLUMN hardware_info TEXT DEFAULT '{}'")
            except Exception:
                pass

        conn.commit()


def insert_daily_record(device_id: str, record: Dict[str, Any]) -> None:
    """Upsert a compact daily record synchronized from a monitoring machine."""
    avg_temp = record.get("average_temperature_c")
    if avg_temp is not None:
        try:
            avg_temp = float(avg_temp)
            if avg_temp <= 0.0 or avg_temp > 125.0:
                avg_temp = None
        except (ValueError, TypeError):
            avg_temp = None

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO daily_metrics
            (device_id, day, reading_count, average_temperature_c, average_fan_rpm,
             average_cpu_usage, average_voltage_v, average_cpu_freq_mhz, spike_count,
             throttling_count, data_quality, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id, day) DO UPDATE SET
              reading_count = excluded.reading_count,
              average_temperature_c = excluded.average_temperature_c,
              average_fan_rpm = excluded.average_fan_rpm,
              average_cpu_usage = excluded.average_cpu_usage,
              average_voltage_v = excluded.average_voltage_v,
              average_cpu_freq_mhz = excluded.average_cpu_freq_mhz,
              spike_count = excluded.spike_count,
              throttling_count = excluded.throttling_count,
              data_quality = excluded.data_quality,
              updated_at = excluded.updated_at
            """,
            (
                device_id,
                record["day"],
                record.get("reading_count", 0),
                avg_temp,
                float(record.get("average_fan_rpm", 0.0) or 0.0),
                float(record.get("average_cpu_usage", 0.0) or 0.0),
                float(record.get("average_voltage_v", 1.2) or 1.2),
                float(record.get("average_cpu_freq_mhz", 0.0) or 0.0),
                int(record.get("spike_count", 0) or 0),
                int(record.get("throttling_count", 0) or 0),
                record.get("data_quality", "ACCEPTABLE"),
                record.get("updated_at", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            ),
        )
        conn.commit()


def update_sync_state(
    device_id: str,
    duration_hours: float,
    daily_records: int,
    total_readings: int,
    spike_events: int,
    hardware_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Record synchronization state and device duration metadata."""
    import json
    hw_str = json.dumps(hardware_info or {})
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO sync_state
            (device_id, monitoring_duration_hours, daily_records, total_readings, spike_events, hardware_info, synchronized_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                monitoring_duration_hours = excluded.monitoring_duration_hours,
                daily_records = excluded.daily_records,
                total_readings = excluded.total_readings,
                spike_events = excluded.spike_events,
                hardware_info = excluded.hardware_info,
                synchronized_at = excluded.synchronized_at
            """,
            (
                device_id,
                float(duration_hours or 0.0),
                int(daily_records or 0),
                int(total_readings or 0),
                int(spike_events or 0),
                hw_str,
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def insert_telemetry(
    device_id: str,
    cpu_temp: Optional[float],
    fan_rpm: int = 0,
    cpu_usage: float = 0.0,
    voltage_v: Optional[float] = None,
    cpu_freq_mhz: Optional[float] = None,
    is_spike: bool = False,
    is_throttled: bool = False,
    timestamp: Optional[str] = None,
) -> None:
    """Store high-resolution telemetry point with data validation."""
    if not timestamp:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Sensor validation: reject invalid temperatures (zero/negative/extreme)
    valid_temp: Optional[float] = None
    if cpu_temp is not None:
        try:
            t = float(cpu_temp)
            if 5.0 <= t <= 125.0:
                valid_temp = t
        except (ValueError, TypeError):
            valid_temp = None

    valid_v = 1.2
    if voltage_v is not None:
        try:
            v = float(voltage_v)
            if 0.3 <= v <= 3.5:
                valid_v = v
        except (ValueError, TypeError):
            pass

    valid_f = 0.0
    if cpu_freq_mhz is not None:
        try:
            f = float(cpu_freq_mhz)
            if 0.5 <= f < 100.0:
                f *= 1000.0  # Scale GHz to MHz
            if 0.0 <= f <= 10000.0:
                valid_f = f
        except (ValueError, TypeError):
            pass

    with sqlite3.connect(DB_FILE) as conn:
        conn.execute(
            """
            INSERT INTO telemetry
            (device_id, cpu_temp, fan_rpm, cpu_usage, voltage_v, cpu_freq_mhz, is_spike, is_throttled, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                device_id,
                valid_temp,
                int(fan_rpm or 0),
                float(max(0.0, min(100.0, cpu_usage or 0.0))),
                valid_v,
                valid_f,
                bool(is_spike or (valid_temp is not None and valid_temp >= 85.0)),
                bool(is_throttled),
                timestamp,
            ),
        )
        conn.commit()


def get_device_metrics(device_id: str) -> Dict[str, Any]:
    """Return validated aggregate metrics and prognostic inputs for a device."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()

        # Check daily_metrics first
        cursor.execute(
            """
            SELECT day, reading_count, average_temperature_c, average_fan_rpm,
                   average_cpu_usage, average_voltage_v, average_cpu_freq_mhz,
                   spike_count, throttling_count, updated_at
            FROM daily_metrics
            WHERE device_id = ?
            ORDER BY day ASC
            """,
            (device_id,),
        )
        daily_rows = cursor.fetchall()

        if daily_rows:
            # Query duration and hardware info from sync state
            cursor.execute(
                "SELECT monitoring_duration_hours, hardware_info, synchronized_at FROM sync_state WHERE device_id = ?",
                (device_id,),
            )
            sync_row = cursor.fetchone()
            op_hours = sync_row[0] if sync_row else len(daily_rows) * 24.0

            total_spikes = sum(row[7] for row in daily_rows)
            total_throttles = sum(row[8] for row in daily_rows)
            total_readings = sum(row[1] for row in daily_rows)

            # Valid temperature aggregation (strictly ignoring null and <= 0)
            valid_temps = [row[2] for row in daily_rows if row[2] is not None and row[2] > 0.0]
            avg_temp = (sum(valid_temps) / len(valid_temps)) if valid_temps else None
            peak_temp = max(valid_temps) if valid_temps else None
            min_temp = min(valid_temps) if valid_temps else None

            voltages = [row[5] for row in daily_rows if row[5] is not None and row[5] > 0.0]
            avg_v = (sum(voltages) / len(voltages)) if voltages else 1.2

            cpus = [row[4] for row in daily_rows if row[4] is not None]
            avg_cpu = (sum(cpus) / len(cpus)) if cpus else 0.0

            freqs = [row[6] for row in daily_rows if row[6] is not None and row[6] > 0.0]
            avg_freq = (sum(freqs) / len(freqs)) if freqs else None

            last_ts = daily_rows[-1][9] if daily_rows else None

            # Build historical intervals for Miner's cumulative rule integration
            intervals: List[Dict[str, Any]] = []
            interval_hours = max(0.1, op_hours / max(1, len(daily_rows)))
            for row in daily_rows:
                if row[2] is not None and row[2] > 0.0:
                    intervals.append({
                        "duration_hours": interval_hours,
                        "temperature_c": row[2],
                        "voltage_v": row[5] or 1.2,
                        "day": row[0],
                    })

            data_quality = "EXCELLENT" if avg_temp is not None and total_readings >= 30 else ("DEGRADED" if avg_temp is not None else "INSUFFICIENT")

            return {
                "device_id": device_id,
                "avg_temp": avg_temp,
                "peak_temp": peak_temp,
                "min_temp": min_temp,
                "total_spikes_over_85": total_spikes,
                "total_spikes": total_spikes,
                "throttling_events": total_throttles,
                "operating_hours": round(float(op_hours), 2),
                "daily_readings": total_readings,
                "reading_count": total_readings,
                "voltage_v": round(float(avg_v), 3),
                "cpu_usage_avg": round(float(avg_cpu), 2),
                "frequency_mhz_avg": round(float(avg_freq), 1) if avg_freq else None,
                "last_timestamp": last_ts,
                "data_quality": data_quality,
                "historical_intervals": intervals,
            }

        # Otherwise check live telemetry table
        cursor.execute(
            """
            SELECT COUNT(*), AVG(cpu_temp), MAX(cpu_temp), MIN(cpu_temp),
                   SUM(is_spike), SUM(is_throttled), AVG(cpu_usage), AVG(voltage_v), AVG(cpu_freq_mhz),
                   MIN(timestamp), MAX(timestamp)
            FROM telemetry
            WHERE device_id = ?
            """,
            (device_id,),
        )
        row = cursor.fetchone()
        total_records = row[0] or 0

        if total_records == 0:
            return {
                "device_id": device_id,
                "avg_temp": None,
                "peak_temp": None,
                "min_temp": None,
                "total_spikes_over_85": 0,
                "total_spikes": 0,
                "throttling_events": 0,
                "operating_hours": 0.0,
                "daily_readings": 0,
                "reading_count": 0,
                "voltage_v": 1.2,
                "cpu_usage_avg": 0.0,
                "frequency_mhz_avg": None,
                "last_timestamp": None,
                "data_quality": "INSUFFICIENT",
                "historical_intervals": [],
            }

        # Calculate valid average temperature ignoring 0.0 and nulls
        cursor.execute(
            "SELECT AVG(cpu_temp), MAX(cpu_temp), MIN(cpu_temp) FROM telemetry WHERE device_id = ? AND cpu_temp > 0.0",
            (device_id,),
        )
        t_row = cursor.fetchone()
        avg_temp = t_row[0] if (t_row and t_row[0] is not None) else None
        peak_temp = t_row[1] if (t_row and t_row[1] is not None) else None
        min_temp = t_row[2] if (t_row and t_row[2] is not None) else None

        total_spikes = row[4] or 0
        total_throttles = row[5] or 0
        avg_cpu = row[6] or 0.0
        avg_v = row[7] or 1.2
        avg_freq = row[8] if (row[8] and row[8] > 0) else None

        first_ts, last_ts = row[9], row[10]
        operating_hours = 0.0
        if first_ts and last_ts:
            try:
                t1 = datetime.datetime.fromisoformat(first_ts.replace("Z", "+00:00")).timestamp()
                t2 = datetime.datetime.fromisoformat(last_ts.replace("Z", "+00:00")).timestamp()
                operating_hours = max(0.0, (t2 - t1) / 3600.0)
            except (ValueError, TypeError):
                operating_hours = total_records / 60.0
        else:
            operating_hours = total_records / 60.0

        data_quality = "EXCELLENT" if avg_temp is not None and total_records >= 30 else ("DEGRADED" if avg_temp is not None else "INSUFFICIENT")

        return {
            "device_id": device_id,
            "avg_temp": avg_temp,
            "peak_temp": peak_temp,
            "min_temp": min_temp,
            "total_spikes_over_85": total_spikes,
            "total_spikes": total_spikes,
            "throttling_events": total_throttles,
            "operating_hours": round(float(operating_hours), 2),
            "daily_readings": total_records,
            "reading_count": total_records,
            "voltage_v": round(float(avg_v), 3),
            "cpu_usage_avg": round(float(avg_cpu), 2),
            "frequency_mhz_avg": round(float(avg_freq), 1) if avg_freq else None,
            "last_timestamp": last_ts,
            "data_quality": data_quality,
            "historical_intervals": [],
        }


def get_device_rul(device_id: str) -> Dict[str, Any]:
    """Return the latest aggregate metrics for RUL evaluation."""
    return get_device_metrics(device_id)


def get_device_spike_summary(device_id: str, hours: int = 24) -> Dict[str, Any]:
    """Retrieve thermal spike events and peak temperatures within a trailing time window."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        time_threshold = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)).isoformat()
        cursor.execute(
            """
            SELECT COUNT(*), AVG(cpu_temp), SUM(is_spike), MAX(cpu_temp)
            FROM telemetry
            WHERE device_id = ? AND timestamp >= ? AND cpu_temp > 0.0
            """,
            (device_id, time_threshold),
        )
        row = cursor.fetchone()
        return {
            "device_id": device_id,
            "hours": hours,
            "total_readings": row[0] or 0,
            "avg_cpu_temp": round(row[1], 2) if row[1] is not None else None,
            "total_spike_count": row[2] or 0,
            "max_temperature": round(row[3], 2) if row[3] is not None else None,
        }


if __name__ == "__main__":
    init_db()
    print("Prediction SQLite database initialized successfully.")
