import sqlite3
import time
from typing import List, Dict, Any

class VoltageHistoryStore:
    """
    Persistent Time-Series Database for past history of System Voltages (VCore, 12V, 5V, 3.3V, Battery),
    Power Consumption (Watts), Utility Scores, and Transients.
    """

    def __init__(self, db_path: str = "voltage_history.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voltage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    vcore_volts REAL NOT NULL,
                    v12_volts REAL NOT NULL,
                    v5_volts REAL NOT NULL,
                    v33_volts REAL NOT NULL,
                    battery_volts REAL NOT NULL,
                    power_watts REAL NOT NULL,
                    utility_score REAL NOT NULL,
                    voltage_status TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    device_profile TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_volts_timestamp ON voltage_history(timestamp)")
            conn.commit()

    def record_entry(self, data: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO voltage_history (
                    timestamp, vcore_volts, v12_volts, v5_volts, v33_volts,
                    battery_volts, power_watts, utility_score, voltage_status,
                    action_taken, device_profile
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("timestamp", time.time()),
                data.get("vcore_volts", 1.15),
                data.get("v12_volts", 12.0),
                data.get("v5_volts", 5.0),
                data.get("v33_volts", 3.3),
                data.get("battery_volts", 0.0),
                data.get("power_watts", 0.0),
                data.get("utility_score", 1.0),
                data.get("voltage_status", "NORMAL"),
                data.get("action_taken", "STABLE_PASSIVE"),
                data.get("device_profile", "laptop")
            ))
            conn.commit()

    def get_recent_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM voltage_history ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_summary_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*), AVG(vcore_volts), MIN(vcore_volts), MAX(vcore_volts), AVG(power_watts), MAX(power_watts)
                FROM voltage_history
            """)
            row = cursor.fetchone()
            count, avg_v, min_v, max_v, avg_w, max_w = row
            return {
                "total_records": count or 0,
                "avg_vcore_volts": round(avg_v, 3) if avg_v else 1.15,
                "min_vcore_volts": round(min_v, 3) if min_v else 1.15,
                "max_vcore_volts": round(max_v, 3) if max_v else 1.15,
                "avg_power_watts": round(avg_w, 1) if avg_w else 0.0,
                "peak_power_watts": round(max_w, 1) if max_w else 0.0
            }
