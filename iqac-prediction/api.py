"""FastAPI Prediction, Prognostics, and Telemetry Service."""

from __future__ import annotations

import datetime
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from database import (
    DB_FILE,
    get_device_metrics,
    get_device_spike_summary,
    init_db,
    insert_daily_record,
    insert_telemetry,
)
from ml_engine import HybridRULEngine, MLRULEngine
from rul_engine import (
    COMPONENT_PROFILES,
    PhysicsRULEngine,
    PredictionMethod,
    RULPredictionResult,
    compute_rul,
    get_profile,
)

app = FastAPI(
    title="IQAC CPU Prognostics & Diagnostics Platform",
    description="Scientifically defensible Remaining Useful Life prediction and agentic telemetry diagnostics.",
    version="2.1.0",
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize database on startup
init_db()

# Optional API authentication key
API_KEY: Optional[str] = os.getenv("IQAC_API_KEY")


def verify_auth(x_api_key: Optional[str] = None, authorization: Optional[str] = None) -> None:
    if not API_KEY:
        return
    token = x_api_key
    if not token and authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")


class TelemetryPayload(BaseModel):
    device_id: str
    cpu_temp: Optional[float] = Field(None, description="CPU temperature in Celsius; rejected if <=0 or >125")
    fan_rpm: int = Field(0, ge=0, le=25000)
    cpu_usage: float = Field(0.0, ge=0.0, le=100.0)
    voltage_v: Optional[float] = Field(None, ge=0.2, le=3.5)
    cpu_freq_mhz: Optional[float] = Field(None, ge=0.0, le=10000.0)
    is_spike: bool = False
    is_throttled: bool = False
    timestamp: Optional[str] = None


class DailyRecordPayload(BaseModel):
    device_id: str
    day: str
    reading_count: int = Field(..., ge=1)
    average_temperature_c: Optional[float] = None
    average_fan_rpm: float = Field(0.0, ge=0.0)
    average_cpu_usage: float = Field(0.0, ge=0.0, le=100.0)
    average_voltage_v: Optional[float] = Field(1.2, ge=0.2, le=3.5)
    average_cpu_freq_mhz: Optional[float] = Field(0.0, ge=0.0)
    spike_count: int = Field(0, ge=0)
    throttling_count: int = Field(0, ge=0)
    data_quality: Optional[str] = "ACCEPTABLE"
    updated_at: str


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view(request: Request):
    """Serve the interactive glassmorphic prognostic dashboard."""
    return templates.TemplateResponse(request, "dashboard.html")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "iqac-prognostics-engine",
        "version": "2.1.0",
        "supported_modes": ["physics", "ml", "hybrid"],
        "profiles": list(COMPONENT_PROFILES.keys()),
    }


@app.get("/devices")
async def list_devices() -> List[str]:
    """List all registered devices in the prediction database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT device_id FROM (
                SELECT device_id FROM sync_state
                UNION
                SELECT device_id FROM daily_metrics
                UNION
                SELECT device_id FROM telemetry
            )
            """
        )
        return [row[0] for row in cursor.fetchall()]


@app.get("/components/profiles")
async def list_component_profiles():
    """List available component reliability parameter profiles."""
    return {
        name: {
            "name": p.name,
            "base_life_hours": p.base_life_hours,
            "ref_temp_c": p.ref_temp_c,
            "max_operating_temp_c": p.max_operating_temp_c,
            "ref_voltage_v": p.ref_voltage_v,
            "activation_energy_ev": p.activation_energy_ev,
            "is_calibrated": p.is_calibrated,
            "description": p.description,
        }
        for name, p in COMPONENT_PROFILES.items()
    }


@app.post("/daily-record")
async def receive_daily_record(
    payload: DailyRecordPayload,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    verify_auth(x_api_key, authorization)
    insert_daily_record(payload.device_id, payload.model_dump())
    return {"status": "success", "message": f"Daily record for {payload.day} synchronized"}


@app.post("/telemetry")
async def receive_telemetry(
    payload: TelemetryPayload,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    verify_auth(x_api_key, authorization)
    try:
        timestamp = payload.timestamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
        insert_telemetry(
            device_id=payload.device_id,
            cpu_temp=payload.cpu_temp,
            fan_rpm=payload.fan_rpm,
            cpu_usage=payload.cpu_usage,
            voltage_v=payload.voltage_v,
            cpu_freq_mhz=payload.cpu_freq_mhz,
            is_spike=payload.is_spike,
            is_throttled=payload.is_throttled,
            timestamp=timestamp,
        )
        return {"status": "success", "message": "Telemetry point validated and saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/devices/{device_id}/metrics")
async def device_metrics_endpoint(device_id: str):
    """Retrieve raw and aggregated metrics for a device."""
    return get_device_metrics(device_id)


@app.get("/devices/{device_id}/rul")
async def device_rul(
    device_id: str,
    mode: str = Query("physics", pattern="^(physics|ml|hybrid)$"),
    profile: str = Query("desktop"),
):
    """Compute scientifically defensible Remaining Useful Life (RUL).

    Supports:
      - mode=physics (Arrhenius + Miner's cumulative rule)
      - mode=ml (Supervised regression if calibrated model exists)
      - mode=hybrid (Fused physics + ML with disagreement tracking)
    """
    metrics = get_device_metrics(device_id)
    comp_profile = get_profile(profile)

    avg_temp = metrics["avg_temp"]
    total_spikes = metrics["total_spikes_over_85"]
    operating_hours = metrics["operating_hours"]
    voltage_v = metrics.get("voltage_v")
    cpu_usage_avg = metrics.get("cpu_usage_avg")
    frequency_mhz_avg = metrics.get("frequency_mhz_avg")
    throttling_events = metrics.get("throttling_events", 0)
    reading_count = metrics.get("reading_count", 0)
    last_ts = metrics.get("last_timestamp")
    historical_intervals = metrics.get("historical_intervals")

    if mode == "ml":
        ml_engine = MLRULEngine()
        result_obj = ml_engine.predict(
            avg_temp=avg_temp,
            total_spikes_over_85=total_spikes,
            operating_hours=operating_hours,
            voltage_v=voltage_v,
            cpu_usage_avg=cpu_usage_avg,
            reading_count=reading_count,
            last_timestamp=last_ts,
        )
    elif mode == "hybrid":
        hybrid_engine = HybridRULEngine(profile=comp_profile)
        result_obj = hybrid_engine.evaluate(
            avg_temp=avg_temp,
            total_spikes_over_85=total_spikes,
            operating_hours=operating_hours,
            voltage_v=voltage_v,
            cpu_usage_avg=cpu_usage_avg,
            throttling_events=throttling_events,
            reading_count=reading_count,
            last_timestamp=last_ts,
            historical_intervals=historical_intervals,
        )
    else:
        physics_engine = PhysicsRULEngine(profile=comp_profile)
        result_obj = physics_engine.evaluate(
            avg_temp=avg_temp,
            total_spikes_over_85=total_spikes,
            operating_hours=operating_hours,
            voltage_v=voltage_v,
            cpu_usage_avg=cpu_usage_avg,
            frequency_mhz_avg=frequency_mhz_avg,
            throttling_events=throttling_events,
            reading_count=reading_count,
            last_timestamp=last_ts,
            historical_intervals=historical_intervals,
        )

    response_dict = result_obj.to_dict()
    # Attach telemetry metrics and backward-compatible fields
    response_dict["device_metrics"] = metrics
    response_dict["rul"] = {
        "health_score_pct": response_dict["health_score_pct"],
        "estimated_rul_hours": response_dict["estimated_rul_hours"],
        "risk_level": response_dict["risk_level"],
    }
    return response_dict


@app.get("/devices/{device_id}/history")
async def device_history(device_id: str):
    """Retrieve historical time series of telemetry and cumulative damage for frontend charting."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT day, average_temperature_c, average_cpu_usage, spike_count
            FROM daily_metrics
            WHERE device_id = ?
            ORDER BY day ASC
            """,
            (device_id,),
        )
        rows = cursor.fetchall()

    days = [r[0] for r in rows]
    temps = [r[1] for r in rows]
    cpus = [r[2] for r in rows]
    spikes = [r[3] for r in rows]

    # Calculate cumulative degradation trajectory across days
    cumulative_damages = []
    engine = PhysicsRULEngine()
    cum_hours = 0.0
    for i in range(len(days)):
        t = temps[i] if temps[i] is not None and temps[i] > 0 else 40.0
        cum_hours += 24.0
        tot_spikes = sum(spikes[: i + 1])
        res = engine.evaluate(avg_temp=t, total_spikes_over_85=tot_spikes, operating_hours=cum_hours)
        cumulative_damages.append(res.degradation_percentage or 0.0)

    return {
        "device_id": device_id,
        "days": days,
        "temperatures": temps,
        "cpu_usages": cpus,
        "spike_counts": spikes,
        "cumulative_damages": cumulative_damages,
    }


@app.get("/devices/{device_id}/diagnose")
async def diagnose_device(device_id: str):
    """Run agentic diagnostic reasoning on a device's telemetry."""
    from agent import run_diagnostic_agent
    report = run_diagnostic_agent(device_id)
    return {
        "device_id": device_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "diagnostic_report": report,
    }
