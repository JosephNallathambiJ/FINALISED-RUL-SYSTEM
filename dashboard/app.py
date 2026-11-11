"""Main FastAPI application for Central Predictive Maintenance Control Center."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
from typing import Any, Dict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT_DIR = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT_DIR / "iqac-prediction"
DASHBOARD_DIR = ROOT_DIR / "dashboard"
STATIC_DIR = DASHBOARD_DIR / "static"
TEMPLATES_DIR = DASHBOARD_DIR / "templates"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PRED_DIR) not in sys.path:
    sys.path.insert(0, str(PRED_DIR))

from database import init_db
from dashboard.routes import (
    device_mgr,
    measurement_mgr,
    process_mgr,
    router as api_router,
    set_event_loop,
    stress_tester,
)
from dashboard.websocket import ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and graceful shutdown."""
    # Capture event loop for thread-safe websocket broadcasts
    loop = asyncio.get_running_loop()
    set_event_loop(loop)

    # Initialize SQLite database schema
    try:
        init_db()
        process_mgr.log_buffer.append("Database", "Database tables initialized successfully.")
    except Exception as e:
        process_mgr.log_buffer.append("Database", f"Error initializing database: {e}", level="ERROR")

    # Start automated measurement supervisor in background
    try:
        measurement_mgr.start()
        process_mgr.log_buffer.append(
            "MeasurementManager",
            f"Automated measurement supervisor started (interval: {measurement_mgr.sampling_interval}s).",
        )
    except Exception as e:
        process_mgr.log_buffer.append("MeasurementManager", f"Failed to start measurement supervisor: {e}", level="ERROR")

    yield

    # Teardown / Graceful Shutdown
    process_mgr.log_buffer.append("System", "Dashboard application shutting down...")
    try:
        stress_tester.stop()
    except Exception:
        pass
    try:
        measurement_mgr.stop()
    except Exception:
        pass
    try:
        process_mgr.stop_all()
    except Exception:
        pass


app = FastAPI(
    title="Machine Learning Framework for Physical System Failure Prediction and Reliability Assessment",
    description="Enterprise Platform for CPU Monitoring, Physics-Informed ML RUL Prediction, and Agentic AI Diagnostics",
    version="2.1.0",
    lifespan=lifespan,
)

# Mount static assets
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include API routes
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serve the single-click desktop-friendly Control Center web interface."""
    device_info = device_mgr.get_device_info()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "device_info": device_info,
            "title": "Machine Learning Framework for Physical System Failure Prediction and Reliability Assessment",
        },
    )


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "HEALTHY",
        "role": device_mgr.role,
        "device_id": device_mgr.device_id,
        "is_measuring": measurement_mgr.is_running,
        "is_paused": measurement_mgr.is_paused,
    }


@app.websocket("/ws/live")
async def live_websocket_endpoint(websocket: WebSocket):
    """Real-time bi-directional WebSocket connection for measurements and logs."""
    connected = await ws_manager.connect(websocket)
    if not connected:
        return
    try:
        # Send initial snapshot immediately upon connect
        latest_meas = measurement_mgr.get_latest_snapshot()
        if latest_meas:
            await websocket.send_json({"type": "measurement", "data": latest_meas})
        
        # Send recent logs
        recent_logs = process_mgr.log_buffer.get_logs(limit=25)
        await websocket.send_json({"type": "initial_logs", "data": recent_logs})

        while True:
            # Keep socket alive and handle any commands sent from frontend
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=port, reload=False)

