import argparse
import uvicorn
from agent import run_diagnostic_agent
from database import init_db

def main():
    parser = argparse.ArgumentParser(description="Hardware Diagnostics & Prognostics AI Agent")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI telemetry receiver server")
    parser.add_argument("--diagnose", type=str, help="Run diagnostic agent for a specific device_id", metavar="DEVICE_ID")
    parser.add_argument("--collect-url", type=str, help="Poll a monitoring agent at this URL")
    parser.add_argument("--device-id", type=str, help="Device ID to assign collected telemetry")
    parser.add_argument("--interval", type=float, default=5.0, help="Telemetry polling interval in seconds")
    parser.add_argument("--port", type=int, default=8000, help="Port for the FastAPI server")
    
    args = parser.parse_args()
    
    # Ensure DB is initialized
    init_db()
    
    if args.collect_url:
        if not args.device_id:
            parser.error("--device-id is required with --collect-url")
        from monitor_client import collect_daily_records
        collect_daily_records(args.collect_url, args.device_id, args.interval)
    elif args.serve:
        print(f"Starting FastAPI server on port {args.port}...")
        uvicorn.run("api:app", host="0.0.0.0", port=args.port, reload=True)
    elif args.diagnose:
        print(f"Initializing diagnostic agent for device: {args.diagnose}")
        report = run_diagnostic_agent(args.diagnose)
        print("\n================ FINAL REPORT ================")
        print(report)
        print("==============================================")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
