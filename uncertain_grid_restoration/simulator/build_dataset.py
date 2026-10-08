import os
import pickle
import torch
import numpy as np
from tqdm import tqdm

from simulator.measurement_generator import generate_measurements
from opendssdirect import dss

def build_dataset(scenarios_file, output_file):
    # Initialize OpenDSS simply to extract perfectly aligned topological mappings
    dss("Clear")
    dss('Redirect "dss/ieee33/Master.dss"')
    dss.Solution.Solve()
    
    buses = dss.Circuit.AllBusNames()
    lines = dss.Lines.AllNames()
    
    bus_to_idx = {bus.lower(): i for i, bus in enumerate(buses)}
    
    # Build static topology (undirected graph representation)
    sources = []
    targets = []
    static_r = []
    static_x = []
    edge_to_line = [] # Maps edge index back to line name for measurement injection
    
    for line in lines:
        dss.Circuit.SetActiveElement(f"Line.{line}")
        
        bus1 = dss.Lines.Bus1().split('.')[0].lower()
        bus2 = dss.Lines.Bus2().split('.')[0].lower()
        
        u = bus_to_idx[bus1]
        v = bus_to_idx[bus2]
        
        length = dss.Lines.Length()
        r = dss.Lines.R1() * length
        x = dss.Lines.X1() * length
        
        # Directed edge (u -> v)
        sources.append(u)
        targets.append(v)
        static_r.append(r)
        static_x.append(x)
        edge_to_line.append(line)
        
        # Directed edge (v -> u)
        sources.append(v)
        targets.append(u)
        static_r.append(r)
        static_x.append(x)
        edge_to_line.append(line)
        
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    print(f"Loading true state scenarios from {scenarios_file}...")
    with open(scenarios_file, "rb") as f:
        scenarios = pickle.load(f)
        
    dataset = []
    
    print("Building PyTorch Geometric tensors...")
    for true_state in tqdm(scenarios, desc="Processing Samples"):
        # 1. Generate stochastic measurements with 20% sensor missing rate
        meas = generate_measurements(true_state, missing_probability=0.2)
        
        # 2. Node Features & Node Labels
        # X format: [V_meas_ph1, V_meas_ph2, V_meas_ph3, V_mask]
        X = torch.zeros((len(buses), 4), dtype=torch.float)
        
        V_true = torch.zeros((len(buses), 3), dtype=torch.float)
        theta_true = torch.zeros((len(buses), 3), dtype=torch.float)
        
        for bus in buses:
            idx = bus_to_idx[bus.lower()]
            
            # Input features
            v_meas = meas["voltage"]["values"][bus.lower()]
            v_mask = meas["voltage"]["mask"][bus.lower()]
            
            for ph in range(min(3, len(v_meas))):
                X[idx, ph] = v_meas[ph]
            X[idx, 3] = v_mask
            
            # True Labels
            v_t = true_state["voltage"][bus.lower()]
            a_t = true_state["angle"][bus.lower()]
            for ph in range(min(3, len(v_t))):
                V_true[idx, ph] = v_t[ph]
                theta_true[idx, ph] = a_t[ph]
                
        # 3. Edge Features & Edge Labels
        # E format: [R, X, P_meas(3), Q_meas(3), I_meas(3), mask] = 12 features
        E = torch.zeros((len(sources), 12), dtype=torch.float)
        I_true = torch.zeros((len(sources), 3), dtype=torch.float)
        
        edge_masks = []
        for e_idx in range(len(sources)):
            line = edge_to_line[e_idx]
            
            # Static topology features
            E[e_idx, 0] = static_r[e_idx]
            E[e_idx, 1] = static_x[e_idx]
            
            # Measurement features
            p_meas = meas["power"]["values"][line]["P"]
            q_meas = meas["power"]["values"][line]["Q"]
            i_meas = meas["current"]["values"][line]
            l_mask = meas["power"]["mask"][line]
            
            for ph in range(min(3, len(p_meas))):
                E[e_idx, 2 + ph] = p_meas[ph]
                E[e_idx, 5 + ph] = q_meas[ph]
                E[e_idx, 8 + ph] = i_meas[ph]
            E[e_idx, 11] = l_mask
            edge_masks.append(l_mask)
            
            # True Labels
            i_t = true_state["current"][line]
            for ph in range(min(3, len(i_t))):
                I_true[e_idx, ph] = i_t[ph]
                
        measurement_mask = {
            "node_mask": X[:, 3],
            "edge_mask": torch.tensor(edge_masks, dtype=torch.float)
        }
        
        # 4. Compile Sample
        sample = {
            "node_features": X,
            "edge_index": edge_index,
            "edge_features": E,
            "measurement_mask": measurement_mask,
            "V_true": V_true,
            "theta_true": theta_true,
            "I_true": I_true
        }
        
        dataset.append(sample)
        
    torch.save(dataset, output_file)
    print(f"Saved {len(dataset)} PyTorch dataset samples to {output_file}")

if __name__ == "__main__":
    import os
    import sys
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.getcwd())
    
    os.makedirs("data/state_estimation", exist_ok=True)
    build_dataset("data/state_estimation/scenarios_1k.pkl", "data/state_estimation/dataset_1k.pt")
