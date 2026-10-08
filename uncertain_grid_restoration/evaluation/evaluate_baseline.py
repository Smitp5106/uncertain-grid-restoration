import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np

# Add project root to sys path
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

from simulator.measurement_generator import generate_measurements
from models.gnn_baseline import BaselineGNN
from opendssdirect import dss

def get_topology():
    dss("Clear")
    dss('Redirect "dss/ieee33/Master.dss"')
    dss.Solution.Solve()
    buses = dss.Circuit.AllBusNames()
    lines = dss.Lines.AllNames()
    bus_to_idx = {bus.lower(): i for i, bus in enumerate(buses)}
    
    sources, targets, static_r, static_x, edge_to_line = [], [], [], [], []
    for line in lines:
        dss.Circuit.SetActiveElement(f"Line.{line}")
        bus1 = dss.Lines.Bus1().split('.')[0].lower()
        bus2 = dss.Lines.Bus2().split('.')[0].lower()
        u, v = bus_to_idx[bus1], bus_to_idx[bus2]
        length = dss.Lines.Length()
        r, x = dss.Lines.R1() * length, dss.Lines.X1() * length
        
        sources.extend([u, v])
        targets.extend([v, u])
        static_r.extend([r, r])
        static_x.extend([x, x])
        edge_to_line.extend([line, line])
        
    return buses, bus_to_idx, torch.tensor([sources, targets], dtype=torch.long), static_r, static_x, edge_to_line

def build_data_from_scenario(scenario, missing_prob, topo):
    buses, bus_to_idx, edge_index, static_r, static_x, edge_to_line = topo
    meas = generate_measurements(scenario, missing_probability=missing_prob)
    
    X = torch.zeros((len(buses), 4), dtype=torch.float)
    V_true = torch.zeros((len(buses), 3), dtype=torch.float)
    
    # Scale factors to normalize the data for the GNN
    V_SCALE = 10000.0
    PQ_SCALE = 1000.0
    I_SCALE = 1000.0
    
    for bus in buses:
        idx = bus_to_idx[bus.lower()]
        v_meas = meas["voltage"]["values"][bus.lower()]
        v_mask = meas["voltage"]["mask"][bus.lower()]
        for ph in range(min(3, len(v_meas))):
            X[idx, ph] = v_meas[ph] / V_SCALE
        X[idx, 3] = v_mask
        
        v_t = scenario["voltage"][bus.lower()]
        for ph in range(min(3, len(v_t))):
            V_true[idx, ph] = v_t[ph] / V_SCALE
            
    E = torch.zeros((edge_index.shape[1], 12), dtype=torch.float)
    for e_idx in range(edge_index.shape[1]):
        line = edge_to_line[e_idx]
        E[e_idx, 0] = static_r[e_idx]
        E[e_idx, 1] = static_x[e_idx]
        p_meas = meas["power"]["values"][line]["P"]
        q_meas = meas["power"]["values"][line]["Q"]
        i_meas = meas["current"]["values"][line]
        l_mask = meas["power"]["mask"][line]
        for ph in range(min(3, len(p_meas))):
            E[e_idx, 2 + ph] = p_meas[ph] / PQ_SCALE
            E[e_idx, 5 + ph] = q_meas[ph] / PQ_SCALE
            E[e_idx, 8 + ph] = i_meas[ph] / I_SCALE
        E[e_idx, 11] = l_mask
        
    return X, edge_index, E, V_true

def evaluate_baseline():
    topo = get_topology()
    
    # Load scenarios
    with open("data/state_estimation/scenarios_1k.pkl", "rb") as f:
        scenarios = pickle.load(f)
        
    train_scenarios = scenarios[:800]
    test_scenarios = scenarios[800:]
    
    model = BaselineGNN(node_in_dim=4, edge_in_dim=12, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    mse = nn.MSELoss()
    
    print("Training Baseline GNN on 800 scenarios for 5 epochs with internal scaling...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        for sc in train_scenarios:
            optimizer.zero_grad()
            # Random masking during training to generalize
            missing_prob = np.random.uniform(0.0, 0.8)
            X, edge_index, E, V_true = build_data_from_scenario(sc, missing_prob, topo)
            
            v_hat, _, _ = model(X, edge_index, E)
            loss = mse(v_hat, V_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_scenarios):.6f}")

    print("\n--- Experimental Evaluation ---")
    print("Measurement availability\tV RMSE (Volts)")
    print("-" * 55)
    
    availabilities = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    model.eval()
    with torch.no_grad():
        for avail in availabilities:
            missing_prob = 1.0 - avail
            
            errors = []
            for sc in test_scenarios:
                X, edge_index, E, V_true = build_data_from_scenario(sc, missing_prob, topo)
                v_hat, _, _ = model(X, edge_index, E)
                
                # RMSE calculation
                mse_val = mse(v_hat, V_true).item()
                rmse_volts = np.sqrt(mse_val) * 10000.0 # Denormalize
                errors.append(rmse_volts)
                
            mean_rmse = np.mean(errors)
            print(f"{int(avail*100)}%\t\t\t\t{mean_rmse:.2f}")

if __name__ == "__main__":
    evaluate_baseline()
