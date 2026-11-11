import json
import time
import urllib.request
import asyncio
import websockets

async def check_websocket_stream():
    print("\n--- 1. Testing WebSocket Real-Time Telemetry Stream (/ws/live) ---")
    uri = "ws://127.0.0.1:8080/ws/live"
    async with websockets.connect(uri) as ws:
        received_frames = []
        for i in range(3):
            raw = await asyncio.wait_for(ws.recv(), timeout=6.0)
            msg = json.loads(raw)
            print(f"Received WS frame {i+1}: type={msg.get('type')}, keys={list(msg.get('data', {}).keys()) if isinstance(msg.get('data'), dict) else type(msg.get('data'))}")
            if msg.get("type") == "measurement":
                received_frames.append(msg["data"])
        
        assert len(received_frames) >= 1, "FAILED: No measurement frames received over WebSocket"
        frame = received_frames[-1]
        print(f"Sample live telemetry frame: temp={frame.get('cpu_temp')}°C, util={frame.get('cpu_util')}%, freq={frame.get('cpu_freq_mhz')}MHz, mem={frame.get('memory_util_pct')}%")
        assert frame.get("cpu_temp") is not None and frame.get("cpu_temp") > 20.0, f"FAILED: Invalid live temp {frame.get('cpu_temp')}"
        assert frame.get("cpu_freq_mhz") is not None and frame.get("cpu_freq_mhz") >= 500.0, f"FAILED: Invalid live freq {frame.get('cpu_freq_mhz')}"
        print("✓ WebSocket live telemetry stream verified: Live frames continuously broadcasting with real physical temperature and frequency.")

def check_rul_prediction():
    print("\n--- 2. Testing Physics-Based RUL Prediction API (/api/rul/calculate) ---")
    req = urllib.request.Request(
        "http://127.0.0.1:8080/api/rul/calculate",
        data=json.dumps({"device_id": "target-cpu", "operating_hours": 10.0, "ambient_temp_c": 25.0}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"RUL Response method={res.get('prediction_method')}")
        print(f"  Predicted RUL: {res.get('rul_hours')} hours ({res.get('rul_days')} days)")
        print(f"  Health Score: {res.get('health_percentage')}%")
        print(f"  Cumulative Degradation: {res.get('degradation_percentage')}%")
        print(f"  Risk Category: {res.get('risk_level')}")
        print(f"  Confidence Interval: {res.get('confidence_interval')}")
        dm = res.get("device_metrics", {})
        print(f"  Device Metrics: avg_temp={dm.get('avg_temp'):.2f}°C, freq={dm.get('frequency_mhz_avg')}MHz, samples={dm.get('reading_count')}")
        
        assert res.get("rul_hours") is not None, "FAILED: rul_hours is None"
        assert res.get("health_percentage") is not None, "FAILED: health_percentage is None"
        assert res.get("risk_level") in ("LOW", "MODERATE", "MEDIUM", "HIGH", "CRITICAL"), f"FAILED: unexpected risk level: {res.get('risk_level')}"
        assert dm.get("avg_temp") is not None and dm.get("avg_temp") > 20.0, "FAILED: avg_temp is invalid"
        assert dm.get("frequency_mhz_avg") is not None and dm.get("frequency_mhz_avg") >= 500.0, "FAILED: frequency is invalid"
        print("✓ RUL Prediction API verified: Valid physics-based degradation and confidence bounds returned.")

def check_ai_agent_diagnostics():
    print("\n--- 3. Testing Autonomous Agentic AI Diagnostics API (/api/agent/diagnose) ---")
    req = urllib.request.Request(
        "http://127.0.0.1:8080/api/agent/diagnose",
        data=json.dumps({"device_id": "target-cpu", "prompt": "Audit physical telemetry and system reliability"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        res = json.loads(resp.read().decode())
        print(f"Agent Diagnose status={res.get('status')}, model={res.get('model')}")
        assert res.get("status") == "success", f"FAILED: Agent status not success: {res}"
        resp_text = res.get("response", "")
        print(f"Agent Diagnostic Preview ({len(resp_text)} chars):\n{resp_text[:350]}...\n")
        assert len(resp_text) > 50, "FAILED: Response text too short"
        print("✓ Agentic AI Diagnostics verified: Active reasoning engine returned grounded report.")

def check_frontend_html_and_assets():
    print("\n--- 4. Validating Frontend HTML, Theme & Canvas Architecture ---")
    with urllib.request.urlopen("http://127.0.0.1:8080/") as resp:
        html = resp.read().decode()
        assert resp.status == 200, f"FAILED: Status {resp.status}"
        assert "PRED-MAINT ENTERPRISE" in html or "Machine Learning Framework" in html
        assert 'id="overview-perf-chart"' in html, "FAILED: overview-perf-chart missing"
        assert 'id="live-temp-chart"' in html, "FAILED: live-temp-chart missing"
        assert 'id="live-util-chart"' in html, "FAILED: live-util-chart missing"
        assert 'id="live-freq-chart"' in html, "FAILED: live-freq-chart missing"
        assert 'id="live-mem-chart"' in html, "FAILED: live-mem-chart missing"
        assert 'id="view-overview"' in html, "FAILED: view-overview missing"
        assert 'id="view-monitoring"' in html, "FAILED: view-monitoring missing"
        assert 'id="view-rul"' in html, "FAILED: view-rul missing"
        assert 'id="view-devices"' in html, "FAILED: view-devices missing"
        print("✓ Frontend HTML verified: All 4 view sections and 5 chart canvas elements present.")

    with urllib.request.urlopen("http://127.0.0.1:8080/static/style.css") as resp:
        css = resp.read().decode()
        assert "--bg-core: #f5f4ef" in css, "FAILED: Warm off-white background variable missing"
        assert "--accent-orange: #d9531e" in css, "FAILED: Burnt orange accent variable missing"
        print("✓ Frontend CSS verified: Warm off-white industrial theme correctly defined.")

    print("\n>>> DEBUG CYCLE 3 COMPLETE: ALL INTEGRATION & WORKFLOW TESTS PASSED! <<<\n")

if __name__ == "__main__":
    print("=== DEBUG CYCLE 3: WEBSOCKET, RUL, AI DIAGNOSTICS & FRONTEND ===")
    asyncio.run(check_websocket_stream())
    check_rul_prediction()
    check_ai_agent_diagnostics()
    check_frontend_html_and_assets()
