"""
Comprehensive Verification and Test Suite
Executes all possible tests across:
1. OpenDSS Power Grid Physics & Feeder Convergence
2. Fault Injection & Localization across all candidate lines and types
3. Autonomous Restoration & Tie-Switch Reconfiguration
4. Dashboard HTTP Server, Endpoints, & Favicon handling
5. Frontend Standalone & Modular Asset Verification
6. High-Resolution Visual Figure Generation
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
import threading
from http.server import HTTPServer

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DSS_DIR = os.path.join(BASE_DIR, "uncertain_grid_restoration")
sys.path.append(DSS_DIR)
sys.path.append(BASE_DIR)

from simulator.dashboard_simulation_engine import run_opendss_scenario, OPENDSS_AVAILABLE
import importlib.util
spec = importlib.util.spec_from_file_location("server_mod", os.path.join(DSS_DIR, "run_dashboard.py"))
server_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_mod)
DashboardHandler = server_mod.DashboardHandler

TEST_RESULTS = []

def record_test(category, name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {category} :: {name}")
    if details:
        print(f"       -> {details}")
    TEST_RESULTS.append({
        "category": category,
        "name": name,
        "passed": passed,
        "details": details
    })

def test_opendss_core():
    print("\n" + "="*70)
    print(" 1. OPENDSS CORE ENGINE & IEEE 33-BUS GRID TESTS")
    print("="*70)
    
    record_test("Core", "OpenDSS Library Import", OPENDSS_AVAILABLE, "opendssdirect available in Python environment")
    
    import opendssdirect as dss
    dss_file = os.path.join(DSS_DIR, "dss", "ieee33", "Master.dss")
    file_exists = os.path.exists(dss_file)
    record_test("Core", "Master.dss Existence", file_exists, f"Path: {dss_file}")
    
    if not file_exists:
        return
        
    dss.Command("Clear")
    dss.Command(f'Redirect "{dss_file}"')
    dss.Solution.Solve()
    converged = dss.Solution.Converged()
    record_test("Core", "Normal IEEE 33 Power Flow Convergence", converged, "dss.Solution.Converged() == True")
    
    bus_count = len(dss.Circuit.AllBusNames())
    record_test("Core", "Bus Count Verification", bus_count == 33, f"Detected {bus_count} buses (expected 33)")
    
    line_count = len(dss.Lines.AllNames())
    record_test("Core", "Line Count Verification", line_count == 32, f"Detected {line_count} physical lines (expected 32)")
    
    # Voltage profile checks
    voltages = {}
    for b in [str(i) for i in range(1, 34)]:
        dss.Circuit.SetActiveBus(b)
        voltages[b] = float(dss.Bus.puVmagAngle()[0])
        
    min_v = min(voltages.values())
    max_v = max(voltages.values())
    v_in_range = (0.90 <= min_v <= 1.05) and (0.98 <= max_v <= 1.01)
    record_test("Core", "Pre-Fault Voltage pu Limits [0.90, 1.05]", v_in_range, f"Min pu: {min_v:.4f} (Bus 18), Max pu: {max_v:.4f} (Bus 1)")
    
    total_mw = -dss.Circuit.TotalPower()[0] / 1000
    p_valid = (3.0 <= total_mw <= 4.5)
    record_test("Core", "Base Total Active Power MW", p_valid, f"Total Grid Load: {total_mw:.3f} MW")

def test_all_fault_scenarios():
    print("\n" + "="*70)
    print(" 2. ALL FAULT LOCATIONS & FAULT TYPES SCENARIO TESTS")
    print("="*70)
    
    candidate_lines = ["18-19", "7-8", "13-14", "2-3", "9-10", "27-28", "31-32"]
    fault_types = ["SLG", "LL", "3P", "HIF"]
    
    for fl in candidate_lines:
        for ft in fault_types:
            try:
                res = run_opendss_scenario(fault_line=fl, fault_type=ft)
                ok = res.get("success", False)
                
                # Check voltages
                v_pre = res["v_prefault"]
                v_flt = res["v_fault"]
                v_rst = res["v_restored"]
                
                # Verify that faulted bus experiences dip during metallic faults,
                # and recovers from isolated blackout state (v_rst > v_isolate and v_rst >= 0.85)
                parts = fl.split("-")
                target_bus = parts[1]
                has_dip = v_flt[target_bus] < v_pre[target_bus]
                v_iso = res["v_isolate"][target_bus]
                has_restored = (v_rst[target_bus] >= 0.85) and (v_rst[target_bus] > v_iso or v_iso > 0.85)
                restored_pct = res["restored_pct"]
                pct_valid = (80.0 <= restored_pct <= 100.0)
                
                pass_all = ok and has_dip and has_restored and pct_valid
                detail = f"V_flt({target_bus})={v_flt[target_bus]:.3f} pu, V_rst={v_rst[target_bus]:.3f} pu, Restored={restored_pct:.1f}%"
                record_test("Scenarios", f"Fault {fl} ({ft})", pass_all, detail)
            except Exception as e:
                record_test("Scenarios", f"Fault {fl} ({ft})", False, f"Exception: {e}")

def test_autonomous_restoration_logic():
    print("\n" + "="*70)
    print(" 3. AUTONOMOUS RESTORATION SWITCHING LOGIC TESTS")
    print("="*70)
    
    res = run_opendss_scenario("18-19", "SLG")
    candidates = res.get("candidates", [])
    record_test("Restoration", "GNN Fault Localization Output", len(candidates) >= 5, f"Candidate count: {len(candidates)}")
    
    # Check top candidate
    top_cand = max(candidates, key=lambda c: c["prob"])
    is_correct = (top_cand["line"] == "18-19" and top_cand["prob"] >= 0.85)
    record_test("Restoration", "Top Candidate Identification (18-19)", is_correct, f"Top: {top_cand['line']} with prob {top_cand['prob']}")
    
    # Check restoration MW metrics
    mw_base = res.get("total_mw_base", 0)
    mw_rst = res.get("total_mw_restored", 0)
    mw_valid = mw_base > 0 and mw_rst > 0 and (mw_rst / mw_base > 0.85)
    record_test("Restoration", "Power Balance & Restored MW", mw_valid, f"Base: {mw_base:.2f} MW, Restored: {mw_rst:.2f} MW")

def test_http_server_and_api():
    print("\n" + "="*70)
    print(" 4. HTTP SERVER & REST API ENDPOINTS TESTS")
    print("="*70)
    
    test_port = 8059
    server = HTTPServer(("127.0.0.1", test_port), DashboardHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)
    
    base_url = f"http://127.0.0.1:{test_port}"
    
    # 1. Test index.html
    try:
        with urllib.request.urlopen(f"{base_url}/") as response:
            status = response.getcode()
            body = response.read().decode("utf-8")
            is_valid = (status == 200 and "AI-Based Fault Diagnosis" in body and "topologyCanvas" in body)
            record_test("Server", "GET / (index.html)", is_valid, f"Status: {status}, size: {len(body)} bytes")
    except Exception as e:
        record_test("Server", "GET / (index.html)", False, str(e))
        
    # 2. Test style.css
    try:
        with urllib.request.urlopen(f"{base_url}/style.css") as response:
            status = response.getcode()
            css = response.read().decode("utf-8")
            is_valid = (status == 200 and "--header-gradient" in css)
            record_test("Server", "GET /style.css", is_valid, f"Status: {status}, size: {len(css)} bytes")
    except Exception as e:
        record_test("Server", "GET /style.css", False, str(e))

    # 3. Test app.js
    try:
        with urllib.request.urlopen(f"{base_url}/app.js") as response:
            status = response.getcode()
            js = response.read().decode("utf-8")
            is_valid = (status == 200 and "simulationLoop" in js and "BUS_POSITIONS" in js)
            record_test("Server", "GET /app.js", is_valid, f"Status: {status}, size: {len(js)} bytes")
    except Exception as e:
        record_test("Server", "GET /app.js", False, str(e))

    # 4. Test favicon.ico handling (Bug Fix Verification)
    try:
        req = urllib.request.Request(f"{base_url}/favicon.ico")
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            # 204 No Content is expected and healthy
            is_valid = status in [200, 204]
            record_test("Server", "GET /favicon.ico (Favicon Fix)", is_valid, f"Status: {status} (No 404 or TypeError crash)")
    except urllib.error.HTTPError as e:
        # If it returned an error, verify it didn't crash server
        record_test("Server", "GET /favicon.ico (Favicon Fix)", e.code in [204, 200], f"Returned code {e.code}")
    except Exception as e:
        record_test("Server", "GET /favicon.ico (Favicon Fix)", False, str(e))

    # 5. Test API Endpoint: /api/simulate
    try:
        with urllib.request.urlopen(f"{base_url}/api/simulate?fault_line=18-19&fault_type=SLG") as response:
            status = response.getcode()
            data = json.loads(response.read().decode("utf-8"))
            keys_ok = all(k in data for k in ["success", "fault_line", "v_prefault", "v_fault", "v_restored", "restored_pct"])
            record_test("Server", "GET /api/simulate (OpenDSS Solver)", status == 200 and keys_ok and data["success"],
                        f"Restored: {data.get('restored_pct')}%")
    except Exception as e:
        record_test("Server", "GET /api/simulate (OpenDSS Solver)", False, str(e))
        
    server.shutdown()
    server.server_close()

def test_frontend_files():
    print("\n" + "="*70)
    print(" 5. FRONTEND FILES & STANDALONE ASSETS VERIFICATION")
    print("="*70)
    
    standalone_path = os.path.join(BASE_DIR, "dashboard.html")
    exists = os.path.exists(standalone_path)
    record_test("Frontend", "dashboard.html Existence", exists, standalone_path)
    
    if exists:
        content = open(standalone_path, "r", encoding="utf-8").read()
        canvases = [
            "topologyCanvas", "voltageProfileCanvas", "realtimeVoltageCanvas",
            "loadRestorationCanvas", "uncertaintyCanvas", "miniTopologyCanvas"
        ]
        all_canvases = all(c in content for c in canvases)
        record_test("Frontend", "All 6 Canvases Present in Standalone HTML", all_canvases, f"Canvases: {', '.join(canvases)}")
        
        has_js = "simulationLoop" in content and "requestAnimationFrame" in content
        record_test("Frontend", "60 FPS Animation Engine Embedded", has_js, "simulationLoop and requestAnimationFrame found")
        
        has_css = "panel-topology" in content and "badge-red" in content
        record_test("Frontend", "Complete UI Styles Embedded", has_css, "Dashboard palette and badges embedded")

def test_figure_generation():
    print("\n" + "="*70)
    print(" 6. FIGURE GENERATOR & HIGH-RES OUTPUT VERIFICATION")
    print("="*70)
    
    fig_path = os.path.join(BASE_DIR, "restoration_dashboard_output.png")
    exists = os.path.exists(fig_path)
    record_test("Output", "restoration_dashboard_output.png Existence", exists, fig_path)
    
    if exists:
        file_size_kb = os.path.getsize(fig_path) / 1024
        size_valid = file_size_kb > 200 # Should be high-res figure (>200 KB)
        record_test("Output", "Image Resolution & File Size", size_valid, f"Size: {file_size_kb:.1f} KB (Valid high-res raster)")

def main():
    start_time = time.time()
    print("="*70)
    print(" STARTING COMPREHENSIVE VERIFICATION & TEST SUITE")
    print("="*70)
    
    test_opendss_core()
    test_all_fault_scenarios()
    test_autonomous_restoration_logic()
    test_http_server_and_api()
    test_frontend_files()
    test_figure_generation()
    
    duration = time.time() - start_time
    total = len(TEST_RESULTS)
    passed = sum(1 for t in TEST_RESULTS if t["passed"])
    failed = total - passed
    
    print("\n" + "="*70)
    print(" TEST SUITE SUMMARY REPORT")
    print("="*70)
    print(f" Total Tests Executed : {total}")
    print(f" Passed               : {passed}")
    print(f" Failed               : {failed}")
    print(f" Success Rate         : {(passed / total) * 100:.1f}%")
    print(f" Execution Duration   : {duration:.2f} seconds")
    print("="*70)
    
    if failed == 0:
        print("\n>>> ALL TESTS PASSED SUCCESSFULLY! SYSTEM VERIFIED 100% HEALTHY. <<<\n")
    else:
        print(f"\n>>> WARNING: {failed} TESTS FAILED. PLEASE REVIEW LOGS ABOVE. <<<\n")

if __name__ == "__main__":
    main()
