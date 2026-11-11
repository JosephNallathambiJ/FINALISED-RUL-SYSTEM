import sqlite3
import time
import os
from typing import List, Dict, Any

class TemperatureHistoryStore:
    """
    Persistent Time-Series Database for past history of all temperature readings,
    CPU load, battery levels, actions, and utility evaluations.
    """

    def __init__(self, db_path: str = "temperature_history.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temp_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    temperature_c REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    ram_percent REAL NOT NULL,
                    battery_percent REAL NOT NULL,
                    thermal_velocity REAL NOT NULL,
                    action_taken TEXT NOT NULL,
                    utility_score REAL NOT NULL,
                    sensor_source TEXT NOT NULL,
                    thermal_status TEXT NOT NULL,
                    device_profile TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON temp_history(timestamp)")
            conn.commit()

    def record_entry(self, data: Dict[str, Any]):
        """Persists a new state-action-perception log entry into temperature history database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO temp_history (
                    timestamp, temperature_c, cpu_percent, ram_percent,
                    battery_percent, thermal_velocity, action_taken,
                    utility_score, sensor_source, thermal_status, device_profile
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("timestamp", time.time()),
                data.get("temperature_c", 25.0),
                data.get("cpu_percent", 0.0),
                data.get("ram_percent", 0.0),
                data.get("battery_percent", 100.0),
                data.get("thermal_velocity", 0.0),
                data.get("action_taken", "PASSIVE"),
                data.get("utility_score", 1.0),
                data.get("sensor_source", "simulated"),
                data.get("thermal_status", "NORMAL"),
                data.get("device_profile", "laptop")
            ))
            conn.commit()

    def get_recent_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Returns the N most recent temperature history records."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM temp_history ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_all_temperatures(self) -> List[float]:
        """Returns all past temperature values recorded in history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT temperature_c FROM temp_history ORDER BY id ASC")
            return [row[0] for row in cursor.fetchall()]

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates statistical summary of temperature history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*), AVG(temperature_c), MIN(temperature_c), MAX(temperature_c)
                FROM temp_history
            """)
            row = cursor.fetchone()
            count, avg_temp, min_temp, max_temp = row
            return {
                "total_records": count or 0,
                "avg_temperature_c": round(avg_temp, 2) if avg_temp else 0.0,
                "min_temperature_c": round(min_temp, 2) if min_temp else 0.0,
                "max_temperature_c": round(max_temp, 2) if max_temp else 0.0
            }
