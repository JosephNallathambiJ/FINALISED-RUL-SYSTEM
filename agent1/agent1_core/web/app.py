import asyncio
import os
import json
import time
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from agent1_core.agent import ModelUtilityLearningAgent

app = FastAPI(title="Agent1 - Model-Based Utility-Based Learning AI Agent")

# Global Agent instance
agent_instance: ModelUtilityLearningAgent = None
active_websockets: List[WebSocket] = []

base_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(base_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

# Stress test artificial CPU load generator flag
stress_load_active = False

def init_web_agent(agent: ModelUtilityLearningAgent):
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
    step_data = agent_instance.step()
    return JSONResponse(step_data)

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

@app.post("/api/stress_load")
async def toggle_stress(enable: bool):
    global stress_load_active
    stress_load_active = enable
    if enable:
        # Induce temporary load on temp sensor simulation
        agent_instance.temp_sensor.simulated_temp += 15.0
    return {"stress_load_active": stress_load_active}

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await asyncio.sleep(agent_instance.config.agent_interval_seconds)
            
            # Step the agent loop
            step_data = agent_instance.step()
            
            # Retrieve recent history for charts
            history_data = agent_instance.history_store.get_recent_history(limit=30)
            step_data["history_stream"] = history_data
            
            await websocket.send_text(json.dumps(step_data))
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
