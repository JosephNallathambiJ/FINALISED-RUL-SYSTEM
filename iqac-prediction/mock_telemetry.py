import requests
import random
import time
import datetime

API_URL = "http://127.0.0.1:8000/telemetry"

def generate_mock_data(device_id: str, num_records: int, base_temp: float, failure_mode: bool = False):
    print(f"Generating {num_records} records for {device_id}...")
    for i in range(num_records):
        # Simulate time passing (1 minute per record)
        timestamp = (datetime.datetime.now() - datetime.timedelta(minutes=num_records-i)).isoformat()
        
        if failure_mode and i > num_records * 0.8:
            # Overheating in the last 20% of records
            cpu_temp = base_temp + random.uniform(20.0, 40.0)
            fan_rpm = random.randint(1000, 2000) # Fan might be failing
            is_spike = cpu_temp > 85.0
        else:
            cpu_temp = base_temp + random.uniform(-5.0, 5.0)
            fan_rpm = random.randint(3000, 4500)
            is_spike = cpu_temp > 85.0
            
        payload = {
            "device_id": device_id,
            "cpu_temp": round(cpu_temp, 2),
            "fan_rpm": fan_rpm,
            "is_spike": is_spike,
            "timestamp": timestamp
        }
        
        try:
            requests.post(API_URL, json=payload)
        except requests.exceptions.ConnectionError:
            print("Failed to connect to API. Is the server running? (python main.py --serve)")
            return
            
    print(f"Done generating data for {device_id}.")

if __name__ == "__main__":
    # Ensure server is running before executing this
    # Good device
    generate_mock_data("device_001", num_records=300, base_temp=45.0, failure_mode=False)
    # Failing device
    generate_mock_data("device_002", num_records=300, base_temp=75.0, failure_mode=True)
