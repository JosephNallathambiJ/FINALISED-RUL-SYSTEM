"""WebSocket manager for broadcasting live CPU measurements, logs, and process status with security checks."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect

from control.authentication import security_mgr
from control.server_manager import server_mgr


class ConnectionManager:
    """Manages active browser WebSocket connections and broadcasts real-time telemetry."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.socket_meta: Dict[WebSocket, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> bool:
        """Validate origin and establish authenticated websocket stream."""
        # 1. Validate Origin header
        origin = websocket.headers.get("origin")
        if not security_mgr.validate_origin_header(origin):
            await websocket.close(code=1008, reason="Forbidden origin")
            return False

        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
            client_ip = websocket.client.host if websocket.client else "127.0.0.1"
            cid = client_id or f"ws_{id(websocket):x}"
            self.socket_meta[websocket] = {
                "client_id": cid,
                "ip": client_ip,
                "connected_at": time.time(),
            }
            # Register in server manager connected clients registry
            server_mgr.register_client_heartbeat(
                client_id=cid,
                ip_address=client_ip,
                device_name="Browser Client",
                role="OPERATOR",
            )
        return True

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            if websocket in self.socket_meta:
                meta = self.socket_meta.pop(websocket)
                server_mgr.disconnect_client(meta.get("client_id", ""))

    async def broadcast_json(self, data: Dict[str, Any]) -> None:
        """Broadcast payload to all active web clients; purge disconnected sockets."""
        msg = json.dumps(data)
        async with self._lock:
            to_remove = []
            for ws in self.active_connections:
                try:
                    await ws.send_text(msg)
                except (WebSocketDisconnect, RuntimeError, Exception):
                    to_remove.append(ws)
            for dead_ws in to_remove:
                if dead_ws in self.active_connections:
                    self.active_connections.remove(dead_ws)
                if dead_ws in self.socket_meta:
                    meta = self.socket_meta.pop(dead_ws)
                    server_mgr.disconnect_client(meta.get("client_id", ""))


ws_manager = ConnectionManager()
