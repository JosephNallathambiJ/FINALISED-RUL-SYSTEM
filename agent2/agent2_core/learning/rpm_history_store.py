import sqlite3
import time
from typing import List, Dict, Any

class RpmHistoryStore:
    """
    Persistent Time-Series Database for past history of Fan RPM, PWM Duty Cycle,
    CPU Temperature, Airflow (CFM), Acoustic Noise (dBA), and Utility Scores.
    """

    def __init__(self, db_path: str = "rpm_cooling_history.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rpm_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    current_rpm INTEGER NOT NULL,
                    target_pwm_pct REAL NOT NULL,
                    temperature_c REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    airflow_cfm REAL NOT NULL,
                    noise_dba REAL NOT NULL,
                    utility_score REAL NOT NULL,
                    fan_health TEXT NOT NULL,
                    device_profile TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rpm_timestamp ON rpm_history(timestamp)")
            conn.commit()

    def record_entry(self, data: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rpm_history (
                    timestamp, current_rpm, target_pwm_pct, temperature_c,
                    cpu_percent, airflow_cfm, noise_dba, utility_score,
                    fan_health, device_profile
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("timestamp", time.time()),
                data.get("current_rpm", 0),
                data.get("target_pwm_pct", 0.0),
                data.get("temperature_c", 25.0),
                data.get("cpu_percent", 0.0),
                data.get("airflow_cfm", 0.0),
                data.get("noise_dba", 22.0),
                data.get("utility_score", 1.0),
                data.get("fan_health", "OK"),
                data.get("device_profile", "laptop")
            ))
            conn.commit()

    def get_recent_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM rpm_history ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    def get_summary_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*), AVG(current_rpm), MAX(current_rpm), AVG(noise_dba), MAX(noise_dba)
                FROM rpm_history
            """)
            row = cursor.fetchone()
            count, avg_rpm, max_rpm, avg_dba, max_dba = row
            return {
                "total_records": count or 0,
                "avg_rpm": int(avg_rpm) if avg_rpm else 0,
                "peak_rpm": int(max_rpm) if max_rpm else 0,
                "avg_noise_dba": round(avg_dba, 1) if avg_dba else 22.0,
                "peak_noise_dba": round(max_dba, 1) if max_dba else 22.0
            }
