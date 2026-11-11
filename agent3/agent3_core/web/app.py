import asyncio
import os
import json
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from agent3_core.agent import SystemVoltageAgent

app = FastAPI(title="Agent3 - System Voltage & Electrical Power AI Agent")

agent_instance: SystemVoltageAgent = None
active_websockets: List[WebSocket] = []

base_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(base_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

def init_web_agent(agent: SystemVoltageAgent):
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

@app.get("/api/history")
async def get_history(limit: int = 50):
    if not agent_instance:
        return JSONResponse({"error": "Agent not initialized"}, status_code=500)
    history = agent_instance.history_store.get_recent_history(limit=limit)
    summary = agent_instance.history_store.get_summary_stats()
    learning = agent_instance.learner.update_learning_model()
    return JSONResponse({
        "history": history,
        "summary": summary,
        "learning": learning
    })

@app.post("/api/profile/{profile_name}")
async def set_profile(profile_name: str):
    if not agent_instance:
        return JSONResponse({"error": "Agent not initialized"}, status_code=500)
    if profile_name in agent_instance.config.profiles:
        agent_instance.set_device_profile(profile_name)
        return {"status": "success", "active_profile": profile_name}
    return JSONResponse({"error": "Invalid profile name"}, status_code=400)

@app.post("/api/inject_transient")
async def inject_transient(type: str = "sag"):
    if agent_instance:
        if type == "sag":
            # Induce temporary under-voltage sag
            agent_instance.voltage_sensor.vcore_nom -= 0.35
        elif type == "surge":
            # Induce temporary over-voltage surge
            agent_instance.voltage_sensor.vcore_nom += 0.30
        else:
            # Reset
            agent_instance.voltage_sensor.vcore_nom = agent_instance.profile.vcore_nominal
    return {"status": f"Voltage transient injected: {type}"}

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await asyncio.sleep(agent_instance.config.agent_interval_seconds)
            step_data = agent_instance.step()
            history_data = agent_instance.history_store.get_recent_history(limit=35)
            step_data["history_stream"] = history_data
            await websocket.send_text(json.dumps(step_data))
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
