import json
import time
import urllib.request
import urllib.error
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
pred_dir = root_dir / "iqac-prediction"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(pred_dir) not in sys.path:
    sys.path.insert(0, str(pred_dir))

import agent

def test_cycle1():
    print("=== DEBUG CYCLE 1: AGENTIC AI CORE & QWEN 2.5 INTEGRATION ===")

    # 1. Ollama Model Discovery
    print("\n--- 1. Testing Ollama Model Discovery ---")
    available_models = agent.list_available_ollama_models()
    print(f"Available Ollama Models: {available_models}")
    assert len(available_models) > 0, "FAILED: No Ollama models found"
    assert any("qwen2.5" in m.lower() for m in available_models), f"FAILED: qwen2.5 not found in {available_models}"

    detected_model = agent.detect_ollama_model()
    print(f"Default Detected Model: {detected_model}")
    assert "qwen" in detected_model.lower(), f"FAILED: Expected qwen model, got {detected_model}"

    # Explicit selection test
    explicit = agent.detect_ollama_model(preferred="qwen2.5:3b")
    print(f"Explicit Model Selection (qwen2.5:3b): {explicit}")
    assert explicit == "qwen2.5:3b"
    print("✓ Model discovery and dynamic selection verified.")

    # 2. Diagnostic Tools Execution
    print("\n--- 2. Testing Diagnostic Tools ---")
    m_raw = agent.tool_get_spike_metrics("target-cpu")
    metrics = json.loads(m_raw)
    print(f"tool_get_spike_metrics: avg_temp={metrics.get('avg_temp')}, freq={metrics.get('frequency_mhz_avg')}")
    assert "avg_temp" in metrics

    agents_st = agent.tool_get_multi_agent_status()
    print(f"tool_get_multi_agent_status: {list(agents_st.keys())}")
    assert "agent1_thermal_learner" in agents_st
    assert "agent2_fan_controller" in agents_st
    assert "agent3_voltage_guard" in agents_st
    assert "agent4_governor" in agents_st

    rul_raw = agent.tool_compute_rul(json.dumps({"avg_temp": 55.0, "total_spikes": 1, "operating_hours": 5.0}))
    rul_data = json.loads(rul_raw)
    print(f"tool_compute_rul: hours={rul_data.get('rul_hours')}, health={rul_data.get('health_percentage')}%")
    assert rul_data.get("rul_hours") is not None

    comp = agent.tool_compare_historical_measurements("target-cpu")
    print(f"tool_compare_historical_measurements: samples={comp.get('samples_analyzed')}, drift={comp.get('thermal_drift_detected')}")
    assert "drift_assessment" in comp

    recs = agent.tool_recommend_maintenance("LOW")
    print(f"tool_recommend_maintenance: actions count={len(recs.get('maintenance_actions', []))}")
    assert len(recs["maintenance_actions"]) >= 1
    print("✓ All 6 diagnostic tools successfully executed on real system data.")

    # 3. Full ReAct Workflow with Qwen 2.5
    print("\n--- 3. Testing Full ReAct Workflow with Qwen 2.5 ---")
    t0 = time.time()
    react_res = agent.run_react_diagnostic_workflow(
        device_id="target-cpu",
        prompt="What is the current condition of my CPU?",
        model_name="qwen2.5-coder:latest",
    )
    duration = time.time() - t0
    print(f"ReAct Workflow completed in {duration:.2f}s:")
    print(f"  Model Used: {react_res.get('model_used')}")
    print(f"  Ollama Connected: {react_res.get('ollama_connected')}")
    print(f"  Tool Trace Steps: {len(react_res.get('tool_trace', []))}")
    for step in react_res.get("tool_trace", []):
        print(f"    - Step {step.get('step')}: {step.get('tool')} ({step.get('execution_time_ms')}ms)")

    assert react_res.get("status") == "success"
    assert react_res.get("ollama_connected") is True
    assert "qwen" in react_res.get("model_used").lower()
    assert len(react_res.get("tool_trace", [])) >= 3
    assert len(react_res.get("response", "")) > 100
    sf = react_res.get("structured_findings", {})
    assert len(sf.get("summary", "")) > 10
    assert len(sf.get("issues", "")) > 10
    assert len(sf.get("recommendations", "")) > 10
    print(f"  Summary Preview: {sf.get('summary')[:100]}...")
    print(f"  Response Preview:\n{react_res.get('response')[:250]}...\n")
    print("✓ Full ReAct workflow grounded in live tools passed.")

    # 4. REST Endpoints via HTTP
    print("\n--- 4. Validating Agent REST Endpoints (/api/agent/*) ---")
    with urllib.request.urlopen("http://127.0.0.1:8080/api/agent/status") as resp:
        st = json.loads(resp.read().decode())
        print(f"GET /api/agent/status: ollama_available={st.get('ollama_available')}, active_model={st.get('active_model')}, models={st.get('available_models')}")
        assert st.get("ollama_available") is True
        assert "qwen" in st.get("active_model", "").lower()
        assert len(st.get("tools_connected", [])) >= 4

    with urllib.request.urlopen("http://127.0.0.1:8080/api/agent/models") as resp:
        models_data = json.loads(resp.read().decode())
        print(f"GET /api/agent/models: {models_data.get('models')}")
        assert "qwen2.5-coder:latest" in models_data.get("models", [])

    req = urllib.request.Request(
        "http://127.0.0.1:8080/api/agent/diagnose",
        data=json.dumps({"prompt": "Are there any abnormal thermal patterns?", "model": "qwen2.5-coder:latest"}).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        diag = json.loads(resp.read().decode())
        print(f"POST /api/agent/diagnose: status={diag.get('status')}, model_used={diag.get('model_used')}, trace_len={len(diag.get('tool_trace', []))}")
        assert diag.get("status") == "success"
        assert len(diag.get("tool_trace", [])) >= 3
        assert len(diag.get("response", "")) > 50

    print("\n>>> DEBUG CYCLE 1 COMPLETE: ALL TESTS PASSED SUCCESSFULLY! <<<\n")

if __name__ == "__main__":
    test_cycle1()

