from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


class AgentMLRuntime:
    """Load the saved joblib model from the project and run inference on live agent telemetry."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.model_bundle = self._load_model_bundle()
        self.model = self.model_bundle.get("model") if self.model_bundle else None
        self.preprocessor = self.model_bundle.get("preprocessor") if self.model_bundle else None
        self.feature_names = list(self.model_bundle.get("features", self.model_bundle.get("feature_names", []))) if self.model_bundle else []

    def _candidate_paths(self) -> list[Path]:
        candidates = [
            ROOT / self.agent_name / f"{self.agent_name}_ml_model.joblib",
            ROOT / "agent_ml_models" / "iot_failure_model.joblib",
            ROOT / "iot_failure_model.joblib",
        ]
        if self.agent_name in {"agent1", "agent2", "agent3", "agent4"}:
            candidates.insert(0, ROOT / "agent_ml_models" / f"{self.agent_name}_ml_model.joblib")
        return candidates

    def _load_model_bundle(self) -> dict[str, Any] | None:
        for path in self._candidate_paths():
            if path.exists():
                try:
                    bundle = joblib.load(path)
                except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
                    # Monitoring and control remain usable when an optional model dependency is absent.
                    continue
                if isinstance(bundle, dict) and ("model" in bundle or "features" in bundle or "feature_names" in bundle):
                    return bundle
        return None

    def _coalesce(self, *values: Any, default: float = 0.0) -> float:
        for value in values:
            if value is None:
                continue
            try:
                numeric = float(value)
                if math.isnan(numeric):
                    continue
                return numeric
            except (TypeError, ValueError):
                continue
        return float(default)

    def _build_feature_row(self, runtime_state: dict[str, Any]) -> dict[str, Any]:
        telemetry = runtime_state.get("telemetry", {}) if isinstance(runtime_state.get("telemetry"), dict) else {}
        temp_payload = runtime_state.get("temperature", {}) if isinstance(runtime_state.get("temperature"), dict) else {}
        belief = runtime_state.get("belief_state", {}) if isinstance(runtime_state.get("belief_state"), dict) else {}
        device_id = int(self._coalesce(runtime_state.get("step"), 0.0, default=0.0))

        cpu_percent = self._coalesce(telemetry.get("cpu_percent"), runtime_state.get("cpu_percent"), 0.0, default=0.0)
        ram_percent = self._coalesce(telemetry.get("ram_percent"), runtime_state.get("ram_percent"), 0.0, default=0.0)
        battery_percent = self._coalesce(telemetry.get("battery_percent"), runtime_state.get("battery_percent"), 100.0, default=100.0)
        temp_c = self._coalesce(temp_payload.get("temperature_celsius"), temp_payload.get("temperature_c"), telemetry.get("hw_temperature_c"), runtime_state.get("temperature_c"), belief.get("temperature_c"), 35.0, default=35.0)
        process_count = self._coalesce(telemetry.get("process_count"), runtime_state.get("process_count"), 0.0, default=0.0)
        uptime_hours = max(1.0, self._coalesce(runtime_state.get("uptime_hours"), telemetry.get("uptime_hours"), max(1.0, device_id), default=1.0))
        workload = max(1.0, min(5.0, cpu_percent / 25.0))
        network_latency_ms = max(10.0, cpu_percent * 1.5 + process_count * 0.8 + temp_c * 0.7)
        packet_loss_pct = max(0.0, min(100.0, (process_count / 30.0) + max(0.0, temp_c - 60.0) * 0.3))
        error_count = int(max(0, round(abs(cpu_percent - 50.0) / 10.0 + max(0.0, temp_c - 45.0) * 0.5 + (process_count % 8))))

        row = {
            "Device_ID": float(device_id),
            "CPU_Usage (%)": float(cpu_percent),
            "Memory_Usage (%)": float(ram_percent),
            "Battery_Level (%)": float(battery_percent),
            "Network_Latency (ms)": float(network_latency_ms),
            "Packet_Loss (%)": float(packet_loss_pct),
            "Temperature (°C)": float(temp_c),
            "Uptime (hrs)": float(uptime_hours),
            "Workload_Intensity": float(workload),
            "Error_Count": float(error_count),
        }

        if self.feature_names:
            ordered = {}
            for name in self.feature_names:
                ordered[name] = row.get(name, row.get(name.replace(" ", "_"), 0.0))
            return ordered
        return row

    def predict(self, runtime_state: dict[str, Any]) -> dict[str, Any]:
        if self.model_bundle is None or self.model is None:
            return {"model_loaded": False, "agent_name": self.agent_name, "predicted_label": "N/A", "confidence": 0.0, "explanation": "No saved ML model found for this agent."}

        feature_row = pd.DataFrame([self._build_feature_row(runtime_state)])

        if self.preprocessor is not None:
            features = self.preprocessor.transform(feature_row)
        else:
            features = feature_row

        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0] if hasattr(self.model, "predict_proba") else np.array([1.0])
        confidence = float(np.max(probabilities))
        label = self.model.classes_[int(np.argmax(probabilities))] if hasattr(self.model, "classes_") else prediction

        return {
            "model_loaded": True,
            "agent_name": self.agent_name,
            "predicted_label": str(label),
            "confidence": confidence,
            "feature_row": feature_row.iloc[0].to_dict(),
        }
