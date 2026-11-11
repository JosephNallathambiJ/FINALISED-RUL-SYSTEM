import asyncio
import os
import json
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from agent4_core.agent import CpuPerformanceAgent

app = FastAPI(title="Agent4 - CPU Frequency & Task Performance AI Agent")

agent_instance: CpuPerformanceAgent = None
active_websockets: List[WebSocket] = []

base_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(base_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

def init_web_agent(agent: CpuPerformanceAgent):
    global agent_instance
    agent_instance = agent

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_profile": agent_instance.config.active_profile,
        "profiles": agent_instance.config.profiles
    })

@app.get("/api/status")
async def get_status():
    if not agent_instance:
        return JSONResponse({"error": "Agent not initialized"}, status_code=500)
    return JSONResponse(agent_instance.step())

@app.post("/api/profile/{profile_name}")
async def set_profile(profile_name: str):
    if not agent_instance:
        return JSONResponse({"error": "Agent not initialized"}, status_code=500)
    if profile_name in agent_instance.config.profiles:
        agent_instance.set_device_profile(profile_name)
        return {"status": "success", "active_profile": profile_name}
    return JSONResponse({"error": "Invalid profile name"}, status_code=400)

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await asyncio.sleep(agent_instance.config.agent_interval_seconds)
            step_data = agent_instance.step()
            step_data["history_stream"] = agent_instance.history_store.get_recent_history(limit=30)
            await websocket.send_text(json.dumps(step_data))
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
