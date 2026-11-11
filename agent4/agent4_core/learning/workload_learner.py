from typing import Dict, Any
from agent4_core.learning.cpu_history_store import CpuHistoryStore

class WorkloadLearner:
    def __init__(self, history_store: CpuHistoryStore):
        self.history_store = history_store

    def update_learning_model(self) -> Dict[str, Any]:
        history = self.history_store.get_recent_history(limit=100)
        if not history:
            return {"status": "COLLECTING_DATA", "avg_load": 0.0}

        loads = [r["cpu_percent"] for r in history]
        avg_load = sum(loads) / len(loads)
        return {
            "status": "LEARNED_WORKLOAD",
            "avg_load_percent": round(avg_load, 1),
            "samples_analyzed": len(loads)
        }
