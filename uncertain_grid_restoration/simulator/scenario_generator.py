import os
import pickle
import numpy as np
from opendssdirect import dss
from tqdm import tqdm
from simulator.extract_state import extract_ground_truth

def add_ders():
    """Adds standard distributed energy resources (PV and Wind) to the IEEE-33 bus system."""
    ders = [
        {"name": "PV1", "bus": "15", "type": "solar", "nominal_kw": 200},
        {"name": "PV2", "bus": "22", "type": "solar", "nominal_kw": 150},
        {"name": "PV3", "bus": "25", "type": "solar", "nominal_kw": 250},
        {"name": "Wind1", "bus": "30", "type": "wind", "nominal_kw": 300},
        {"name": "Wind2", "bus": "10", "type": "wind", "nominal_kw": 150},
    ]
    for der in ders:
        dss(f"New Generator.{der['name']} bus1={der['bus']} phases=3 kv=12.66 kw={der['nominal_kw']} pf=1.0 model=1")
    
    return ders

def generate_scenarios(num_scenarios=1000, save_path="data/state_estimation/scenarios.pkl"):
    print(f"Adding DERs and generating {num_scenarios} scenarios...")
    
    # Reload fresh circuit
    dss("Clear")
    dss('Redirect "dss/ieee33/Master.dss"')
    
    ders = add_ders()
    dss.Solution.Solve()
    
    # Store nominal loads to allow for independent random scaling per scenario
    nominal_loads = {}
    load_name = dss.Loads.First()
    while load_name > 0:
        name = dss.Loads.Name()
        kw = dss.Loads.kW()
        kvar = dss.Loads.kvar()
        nominal_loads[name] = {'kw': kw, 'kvar': kvar}
        load_name = dss.Loads.Next()
        
    scenarios = []

    for i in tqdm(range(num_scenarios), desc="Simulating"):
        # 1. Perturb Loads
        load_name = dss.Loads.First()
        while load_name > 0:
            name = dss.Loads.Name()
            # Random uniform variation for each load
            load_multiplier = np.random.uniform(0.7, 1.3)
            
            new_kw = nominal_loads[name]['kw'] * load_multiplier
            new_kvar = nominal_loads[name]['kvar'] * load_multiplier
            
            dss.Loads.kW(new_kw)
            dss.Loads.kvar(new_kvar)
            
            load_name = dss.Loads.Next()
            
        # 2. Perturb DERs (Solar & Wind)
        solar_multiplier = np.random.uniform(0.0, 1.0)
        wind_multiplier = np.random.uniform(0.0, 1.2)
        
        gen_name = dss.Generators.First()
        while gen_name > 0:
            name = dss.Generators.Name().lower()
            if "pv" in name:
                mult = solar_multiplier
            elif "wind" in name:
                mult = wind_multiplier
            else:
                mult = 1.0
                
            nom_kw = next(d['nominal_kw'] for d in ders if d['name'].lower() == name)
            dss.Generators.kW(nom_kw * mult)
            gen_name = dss.Generators.Next()
            
        # 3. Run Power Flow
        dss.Solution.Solve()
        
        if not dss.Solution.Converged():
            print(f"Warning: Scenario {i} did not converge.")
            continue
            
        # 4. Extract True State
        true_state = extract_ground_truth()
        scenarios.append(true_state)
        
    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(scenarios, f)
        
    print(f"Saved {len(scenarios)} valid scenarios to {save_path}")

if __name__ == "__main__":
    # Change working directory so relative path works
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    dss("Clear")
    dss('Redirect "dss/ieee33/Master.dss"')
    dss.Solution.Solve()
    
    generate_scenarios(num_scenarios=1000, save_path="data/state_estimation/scenarios_1k.pkl")
