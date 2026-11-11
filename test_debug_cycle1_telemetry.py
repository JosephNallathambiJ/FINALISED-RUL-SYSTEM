import json
import time
import sqlite3
import urllib.request
import urllib.error
from control.measurement_manager import collect_hardware_telemetry_snapshot

def test_cycle1():
    print("=== DEBUG CYCLE 1: HARDWARE TELEMETRY & DUAL DATABASE LOGGING ===")
    
    # 1. Direct hardware sensor read test
    print("\n--- 1. Testing Direct Hardware Telemetry Snapshot ---")
    snap = collect_hardware_telemetry_snapshot()
    print(f"Snapshot result: {json.dumps(snap, indent=2)}")
    
    assert snap["cpu_temp"] is not None, "FAILED: cpu_temp must not be None"
    assert 20.0 <= snap["cpu_temp"] <= 105.0, f"FAILED: cpu_temp {snap['cpu_temp']} out of physical range"
    assert snap["cpu_freq_mhz"] is not None, "FAILED: cpu_freq_mhz must not be None"
    assert snap["cpu_freq_mhz"] >= 500.0, f"FAILED: cpu_freq_mhz {snap['cpu_freq_mhz']} must be in MHz (not 4 MHz)"
    assert 0.0 <= snap["cpu_util"] <= 100.0, f"FAILED: cpu_util {snap['cpu_util']} invalid"
    assert 0.0 <= snap["memory_util_pct"] <= 100.0, f"FAILED: memory_util_pct {snap['memory_util_pct']} invalid"
    print("✓ Hardware sensor extraction passed: Real physical Apple Silicon junction temp and MHz confirmed.")

    # 2. Check initial row counts in databases
    print("\n--- 2. Checking Initial Database Records ---")
    conn_rec = sqlite3.connect("monitoring_records.db")
    c_rec = conn_rec.cursor()
    c_rec.execute("SELECT sum(reading_count), sum(valid_temp_count), sum(valid_freq_count) FROM daily_metrics")
    init_rec_row = c_rec.fetchone()
    init_readings = init_rec_row[0] or 0
    init_valid_temps = init_rec_row[1] or 0
    init_valid_freqs = init_rec_row[2] or 0
    conn_rec.close()

    conn_tel = sqlite3.connect("iqac-prediction/telemetry.db")
    c_tel = conn_tel.cursor()
    c_tel.execute("SELECT count(*) FROM telemetry")
    initial_tel_count = c_tel.fetchone()[0]
    conn_tel.close()
    print(f"Initial counts -> daily_metrics readings: {init_readings} (valid temps: {init_valid_temps}, valid freqs: {init_valid_freqs}), telemetry.db: {initial_tel_count}")

    # 3. Start continuous measurements via API
    print("\n--- 3. Starting Continuous Measurement Loop via REST API ---")
    req = urllib.request.Request(
        "http://127.0.0.1:8080/api/measurements/start",
        data=json.dumps({"interval_seconds": 1.0, "mode": "idle"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        start_res = json.loads(resp.read().decode())
        print(f"Start API response: {start_res}")
        assert start_res["status"].lower() in ("started", "already_running")

    # 4. Wait for collection cycles
    print("Waiting 12 seconds for measurement supervisor ticks...")
    time.sleep(12)

    # 5. Check updated database counts
    print("\n--- 4. Checking Updated Database Records ---")
    conn_rec = sqlite3.connect("monitoring_records.db")
    c_rec = conn_rec.cursor()
    c_rec.execute("SELECT sum(reading_count), sum(valid_temp_count), sum(valid_freq_count) FROM daily_metrics")
    new_rec_row = c_rec.fetchone()
    new_readings = new_rec_row[0] or 0
    new_valid_temps = new_rec_row[1] or 0
    new_valid_freqs = new_rec_row[2] or 0
    conn_rec.close()

    conn_tel = sqlite3.connect("iqac-prediction/telemetry.db")
    c_tel = conn_tel.cursor()
    c_tel.execute("SELECT count(*) FROM telemetry")
    new_tel_count = c_tel.fetchone()[0]
    c_tel.execute("SELECT timestamp, cpu_temp, cpu_freq_mhz, cpu_usage FROM telemetry ORDER BY id DESC LIMIT 3")
    recent_tel = c_tel.fetchall()
    conn_tel.close()

    print(f"New counts -> daily_metrics: readings={new_readings} (+{new_readings - init_readings}), valid_temps={new_valid_temps} (+{new_valid_temps - init_valid_temps}), valid_freqs={new_valid_freqs} (+{new_valid_freqs - init_valid_freqs})")
    print(f"New counts -> telemetry.db: {new_tel_count} (+{new_tel_count - initial_tel_count})")
    print(f"Recent telemetry.db rows: {recent_tel}")

    assert new_readings > init_readings, "FAILED: daily_metrics reading count did not increment!"
    assert new_valid_temps > init_valid_temps, "FAILED: daily_metrics valid_temp_count did not increment!"
    assert new_valid_freqs > init_valid_freqs, "FAILED: daily_metrics valid_freq_count did not increment!"
    assert new_tel_count > initial_tel_count, "FAILED: telemetry.db did not increment!"
    assert recent_tel[0][1] is not None and recent_tel[0][1] > 20.0, "FAILED: Temperature in telemetry.db is invalid"
    assert recent_tel[0][2] is not None and recent_tel[0][2] > 500.0, "FAILED: Frequency in telemetry.db is invalid"
    print("✓ Dual database persistence verified: Continuous logging active in both stores with real temperature and frequency.")

    # 6. Check REST API endpoints
    print("\n--- 5. Validating Telemetry REST API Endpoints ---")
    for endpoint in [
        "/api/system/info",
        "/api/measurements/status",
        "/api/measurements/history?limit=5",
        "/api/monitoring/series?range_window=5m"
    ]:
        with urllib.request.urlopen(f"http://127.0.0.1:8080{endpoint}") as resp:
            data = json.loads(resp.read().decode())
            print(f"✓ {endpoint} returned status 200 (keys: {list(data.keys()) if isinstance(data, dict) else len(data)})")

    print("\n>>> DEBUG CYCLE 1 COMPLETE: ALL TESTS PASSED SUCCESSFULLY! <<<\n")

if __name__ == "__main__":
    test_cycle1()
