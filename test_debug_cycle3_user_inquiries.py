import json
import time
import urllib.request
import urllib.error

USER_QUERIES = [
    "What is the current condition of my CPU?",
    "Are there any abnormal thermal patterns?",
    "What is causing increased CPU stress?",
    "Explain my latest RUL prediction.",
    "What maintenance actions do you recommend?",
    "Compare current measurements with historical measurements."
]

def test_cycle3():
    print("=== DEBUG CYCLE 3: USER INTERACTION, REACT INQUIRIES & CHART THRESHOLDS ===")

    # 1. Verify Chart Thresholds and UI Assets
    print("\n--- 1. Validating UI Threshold Elements & Assets ---")
    with urllib.request.urlopen("http://127.0.0.1:8080/") as resp:
        html = resp.read().decode()
        assert resp.status == 200
        assert 'id="agent-model-select"' in html, "FAILED: agent-model-select missing"
        assert 'id="ai-query-input"' in html, "FAILED: ai-query-input missing"
        assert 'id="ai-tool-trace-list"' in html, "FAILED: ai-tool-trace-list missing"
        assert 'Threshold: 80°C Limit' in html, "FAILED: Temperature threshold badge missing"
        assert 'Threshold: 90% Load' in html, "FAILED: Utilization threshold badge missing"
        print("✓ HTML UI elements verified: Model selector, query input, ReAct tool trace, and threshold badges present.")

    with urllib.request.urlopen("http://127.0.0.1:8080/static/app.js") as resp:
        js = resp.read().decode()
        assert "Threshold: 80°C Warning" in js, "FAILED: Temperature threshold dataset missing in app.js"
        assert "Threshold: 90% High Load" in js, "FAILED: Utilization threshold dataset missing in app.js"
        assert "renderToolTrace" in js, "FAILED: renderToolTrace missing in app.js"
        assert "executeAgenticInvestigation" in js, "FAILED: executeAgenticInvestigation missing in app.js"
        print("✓ Client controller verified: Threshold series and interactive ReAct dispatchers confirmed.")

    with urllib.request.urlopen("http://127.0.0.1:8080/static/style.css") as resp:
        css = resp.read().decode()
        assert ".tool-trace-item" in css, "FAILED: .tool-trace-item missing in style.css"
        assert ".badge-threshold" in css, "FAILED: .badge-threshold missing in style.css"
        assert ".preset-chip" in css, "FAILED: .preset-chip missing in style.css"
        print("✓ Industrial CSS verified: Tool trace styling, preset chips, and threshold styling confirmed.")

    # 2. Test User Inquiries against Local Ollama (Qwen 2.5)
    print("\n--- 2. Testing User Inquiries with ReAct Agent Orchestrator ---")
    for idx, query in enumerate(USER_QUERIES, 1):
        print(f"\n[Inquiry {idx}/6]: \"{query}\"")
        req = urllib.request.Request(
            "http://127.0.0.1:8080/api/agent/diagnose",
            data=json.dumps({
                "device_id": "target-cpu",
                "prompt": query,
                "model": "qwen2.5-coder:latest"
            }).encode(),
            headers={"Content-Type": "application/json"}
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
            dur = time.time() - t0
            print(f"  Status: {data.get('status')}")
            print(f"  Model: {data.get('model_used')} (Latency: {dur:.2f}s)")
            tools_called = [t.get("tool") for t in data.get("tool_trace", [])]
            print(f"  Tools Executed ({len(tools_called)}): {tools_called}")
            resp_text = data.get("response", "")
            print(f"  Response Preview ({len(resp_text)} chars): {resp_text[:180].strip()}...")

            assert data.get("status") == "success", f"Query failed: {query}"
            assert len(tools_called) >= 3, f"Insufficient tools called for query: {query}"
            assert len(resp_text) > 50, f"Response too short for query: {query}"
            assert data.get("structured_findings") is not None, "Missing structured findings"

    print("\n>>> DEBUG CYCLE 3 COMPLETE: ALL USER INQUIRIES & WORKFLOWS VERIFIED! <<<\n")

if __name__ == "__main__":
    test_cycle3()

