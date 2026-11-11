import sqlite3
import time
from typing import List, Dict, Any

class CpuHistoryStore:
    def __init__(self, db_path: str = "cpu_freq_history.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cpu_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    freq_mhz REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    governor_mode TEXT NOT NULL,
                    utility_score REAL NOT NULL,
                    device_profile TEXT NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cpu_timestamp ON cpu_history(timestamp)")
            conn.commit()

    def record_entry(self, data: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cpu_history (
                    timestamp, freq_mhz, cpu_percent, governor_mode, utility_score, device_profile
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data.get("timestamp", time.time()),
                data.get("freq_mhz", 2400.0),
                data.get("cpu_percent", 0.0),
                data.get("governor_mode", "POWERSAVE"),
                data.get("utility_score", 1.0),
                data.get("device_profile", "laptop")
            ))
            conn.commit()

    def get_recent_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cpu_history ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in reversed(rows)]
