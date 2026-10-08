"""
AI-Powered Distribution Network Restoration Dashboard Server
Hosts the animated interactive dashboard and connects live to OpenDSS IEEE 33-Bus solver.
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import threading
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(CURRENT_DIR, "dashboard")
sys.path.append(CURRENT_DIR)

try:
    from simulator.dashboard_simulation_engine import run_opendss_scenario
except ImportError:
    from uncertain_grid_restoration.simulator.dashboard_simulation_engine import run_opendss_scenario

PORT = 8050

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # OpenDSS Power Flow API Endpoint
        if parsed.path == "/api/simulate":
            params = urllib.parse.parse_qs(parsed.query)
            fault_line = params.get("fault_line", ["18-19"])[0]
            fault_type = params.get("fault_type", ["SLG"])[0]
            feeder = params.get("feeder", ["ieee33"])[0]
            
            print(f"[API] Running OpenDSS Power Flow for {feeder.upper()} Line {fault_line} ({fault_type})...")
            try:
                result = run_opendss_scenario(fault_line=fault_line, fault_type=fault_type, feeder=feeder)
            except Exception as e:
                result = {"success": False, "error": str(e)}
                
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        # Handle browser favicon requests quietly
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        # Serve static dashboard files
        return super().do_GET()

    def log_message(self, format, *args):
        # Safe logging
        try:
            msg = format % args
            # Log API calls or errors
            if "/api/simulate" in msg:
                sys.stdout.write(f"[{self.log_date_time_string()}] {msg}\n")
        except Exception:
            pass

def open_browser():
    time.sleep(1.2)
    url = f"http://localhost:{PORT}"
    print(f"[+] Opening browser automatically: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"[-] Could not auto-launch browser: {e}")

def main():
    print("==========================================================================")
    print("  AI-BASED FAULT DIAGNOSIS AND AUTONOMOUS RESTORATION DASHBOARD (OpenDSS) ")
    print("==========================================================================")
    print(f"  • Serving Dashboard Directory : {DASHBOARD_DIR}")
    print(f"  • OpenDSS Engine Status       : Ready (IEEE 33-Bus)")
    print(f"  • Local Dashboard URL         : http://localhost:{PORT}")
    print("==========================================================================")
    print("Press Ctrl+C to stop the dashboard server.\n")

    threading.Thread(target=open_browser, daemon=True).start()

    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[+] Dashboard server stopped.")
        server.server_close()

if __name__ == "__main__":
    main()
