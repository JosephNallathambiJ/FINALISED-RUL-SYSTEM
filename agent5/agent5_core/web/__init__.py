from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Import the supervisor class – we will instantiate a singleton when the app starts
from agent5_core.supervisor import Supervisor
from agent5_core.config import load_config

app = FastAPI(title="Agent5 Supervisor API", version="1.0.0")

# Allow any origin for simplicity (can be locked down later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global supervisor instance – will be created on first request if not already running
_supervisor: Supervisor = None

def get_supervisor() -> Supervisor:
    global _supervisor
    if _supervisor is None:
        cfg = load_config()
        _supervisor = Supervisor(cfg)
        _supervisor.start()
    return _supervisor

@app.get("/api/overview")
def api_overview():
    sup = get_supervisor()
    return sup.get_overview()

@app.get("/api/events")
def api_events(limit: int = 100):
    sup = get_supervisor()
    return sup.get_events(limit=limit)

@app.get("/api/agent/{agent_id}/events")
def api_agent_events(agent_id: str, limit: int = 50):
    sup = get_supervisor()
    return sup.get_agent_events(agent_id, limit=limit)

@app.post("/api/agent/{agent_id}/restart")
def api_restart_agent(agent_id: str):
    sup = get_supervisor()
    if sup.restart_agent(agent_id):
        return {"status": "restarted", "agent": agent_id}
    raise HTTPException(status_code=404, detail="Agent not found")

@app.post("/api/agent/{agent_id}/stop")
def api_stop_agent(agent_id: str):
    sup = get_supervisor()
    if sup.stop_agent(agent_id):
        return {"status": "stopped", "agent": agent_id}
    raise HTTPException(status_code=404, detail="Agent not found")
