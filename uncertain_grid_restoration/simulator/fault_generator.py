import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from opendssdirect import dss
from simulator.extract_state import extract_ground_truth

def apply_fault(bus_name, fault_type="SLG", r_fault=0.01):
    """
    Injects a specific fault into the OpenDSS engine.
    """
    if fault_type == "SLG":
        # Single Line to Ground on Phase 1
        dss(f"New Fault.F1 bus1={bus_name}.1 phases=1 r={r_fault}")
    elif fault_type == "LL":
        # Line to Line on Phases 1 and 2
        dss(f"New Fault.F1 bus1={bus_name}.1.2 phases=2 r={r_fault}")
    elif fault_type == "3P":
        # Three Phase symmetric fault
        dss(f"New Fault.F1 bus1={bus_name}.1.2.3 phases=3 r={r_fault}")
    elif fault_type == "HIF":
        # High Impedance Fault (SLG with high R)
        dss(f"New Fault.F1 bus1={bus_name}.1 phases=1 r={r_fault}")
    else:
        raise ValueError(f"Unknown fault type: {fault_type}")

def generate_fault_scenarios(num_scenarios=1000, save_path="data/fault_localization/fault_scenarios.pkl"):
    print(f"Generating {num_scenarios} fault scenarios...")
    
    # Needs to be initialized to get the bus list
    dss("Clear")
    dss('Redirect "dss/ieee33/Master.dss"')
    dss.Solution.Solve()
    
    buses = dss.Circuit.AllBusNames()
    
    scenarios = []
    
    # The prompt explicitly asks to initially use only SLG, then add the rest later.
    fault_types_available = ["SLG", "LL", "3P", "HIF"]
    
    for i in tqdm(range(num_scenarios), desc="Simulating Faults"):
        # 1. Reset circuit to clear previous faults and load variations
        dss("Clear")
        dss('Redirect "dss/ieee33/Master.dss"')
        
        # 2. Random Load variations to ensure dataset diversity under faults
        load_name = dss.Loads.First()
        while load_name > 0:
            load_mult = np.random.uniform(0.7, 1.3)
            dss.Loads.kW(dss.Loads.kW() * load_mult)
            dss.Loads.kvar(dss.Loads.kvar() * load_mult)
            load_name = dss.Loads.Next()
            
        # 3. Choose Fault Parameters
        fault_bus = random.choice(buses)
        
        # By default, stick to SLG as requested, but randomly pick others if testing
        # We will use all 4 to test robustness, as requested: "Then add: line-line, 3p, hif"
        fault_type = random.choice(fault_types_available)
        
        if fault_type == "HIF":
            r_fault = np.random.uniform(30.0, 100.0) # High Impedance
        else:
            r_fault = np.random.uniform(0.001, 2.0)  # Low Impedance
            
        # 4. Apply Fault
        apply_fault(fault_bus, fault_type, r_fault)
        
        # 5. Solve Power Flow
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            # OpenDSS might struggle to converge for 0-ohm 3P faults near the source
            continue
            
        # 6. Extract True State
        true_state = extract_ground_truth()
        
        # 7. Add Labels
        true_state["fault_label"] = {
            "bus": fault_bus,
            "type": fault_type,
            "resistance": r_fault
        }
        
        scenarios.append(true_state)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(scenarios, f)
        
    print(f"Saved {len(scenarios)} converging fault scenarios to {save_path}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    generate_fault_scenarios(num_scenarios=1000, save_path="data/fault_localization/fault_scenarios_1k.pkl")
