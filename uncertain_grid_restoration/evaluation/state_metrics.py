import os
import sys
import torch
import torch.optim as optim
import pickle
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

from models.uncertainty_gnn import UncertaintyGNN
from evaluation.evaluate_baseline import get_topology, build_data_from_scenario

def validate_uncertainty_dynamic():
    topo = get_topology()
    with open("data/state_estimation/scenarios_1k.pkl", "rb") as f:
        scenarios = pickle.load(f)
        
    train_scenarios = scenarios[:800]
    test_scenarios = scenarios[800:]
    
    model = UncertaintyGNN(node_in_dim=4, edge_in_dim=12, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print("Training Uncertainty GNN with DYNAMIC masking to force epistemic learning...")
    
    model.train()
    for epoch in range(10):
        for sc in train_scenarios:
            optimizer.zero_grad()
            # Random masking between 10% and 90% missing so it learns to rely on the mask
            X, edge_index, E, V_true = build_data_from_scenario(sc, missing_prob=np.random.uniform(0.1, 0.9), topo=topo)
            
            preds = model(X, edge_index, E)
            
            variance = preds["v_sigma"] ** 2
            loss = torch.mean(0.5 * torch.log(variance) + 0.5 * ((V_true - preds["v_mean"]) ** 2) / variance)
            
            loss.backward()
            optimizer.step()

    model.eval()
    
    measured_rmses, measured_nlls, measured_sigmas, measured_coverage = [], [], [], []
    missing_rmses, missing_nlls, missing_sigmas, missing_coverage = [], [], [], []
    
    with torch.no_grad():
        for sc in test_scenarios:
            # 50% missing for evaluation
            X, edge_index, E, V_true = build_data_from_scenario(sc, missing_prob=0.5, topo=topo)
            preds = model(X, edge_index, E)
            
            v_mean = preds["v_mean"].reshape(-1)
            v_sigma = preds["v_sigma"].reshape(-1)
            v_true = V_true.reshape(-1)
            v_mask = X[:, 3].unsqueeze(1).expand(-1, 3).reshape(-1)
            
            measured_idx = (v_mask == 1)
            missing_idx = (v_mask == 0)
            
            for indices, rmses, nlls, sigmas, coverages in [
                (measured_idx, measured_rmses, measured_nlls, measured_sigmas, measured_coverage),
                (missing_idx, missing_rmses, missing_nlls, missing_sigmas, missing_coverage)
            ]:
                if not indices.any(): continue
                
                mean_p = v_mean[indices]
                sig_p = v_sigma[indices]
                true_p = v_true[indices]
                
                rmse = torch.sqrt(torch.mean((mean_p - true_p)**2)).item()
                rmses.append(rmse)
                
                var_p = sig_p ** 2
                nll = torch.mean(0.5 * torch.log(var_p) + 0.5 * ((true_p - mean_p)**2) / var_p).item()
                nlls.append(nll)
                
                sigmas.append(torch.mean(sig_p).item())
                
                lower = mean_p - 1.96 * sig_p
                upper = mean_p + 1.96 * sig_p
                covered = ((true_p >= lower) & (true_p <= upper)).float().mean().item()
                coverages.append(covered)

    print("\n--- Uncertainty Validation Results ---")
    print(f"{'Metric':<25} | {'Sensor Available':<20} | {'Sensor Missing'}")
    print("-" * 65)
    print(f"{'V RMSE (per unit)':<25} | {np.mean(measured_rmses):<20.4f} | {np.mean(missing_rmses):.4f}")
    print(f"{'Negative Log-Likelihood':<25} | {np.mean(measured_nlls):<20.4f} | {np.mean(missing_nlls):.4f}")
    print(f"{'Predicted Sigma (Uncert)':<25} | {np.mean(measured_sigmas):<20.4f} | {np.mean(missing_sigmas):.4f}")
    print(f"{'95% CI Coverage':<25} | {np.mean(measured_coverage)*100:<19.2f}% | {np.mean(missing_coverage)*100:.2f}%")

if __name__ == "__main__":
    validate_uncertainty_dynamic()
