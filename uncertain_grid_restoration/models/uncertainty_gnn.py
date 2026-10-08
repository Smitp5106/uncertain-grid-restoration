import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear

class UncertaintyGNN(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64):
        super(UncertaintyGNN, self).__init__()
        
        self.node_emb = Linear(node_in_dim, hidden_dim)
        
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        
        # Node-level MLP (Outputs 3 phases * 2 params * 2 variables = 12 values)
        # [V_mean(3), V_sigma(3), theta_mean(3), theta_sigma(3)]
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 12) 
        )
        
        # Edge-level MLP (Outputs 3 phases * 2 params = 6 values)
        # [I_mean(3), I_sigma(3)]
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)
        )
        
        # Softplus ensures standard deviation is strictly positive
        self.softplus = nn.Softplus()

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.node_emb(x))
        h = F.relu(self.conv1(h, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        h = self.conv3(h, edge_index, edge_attr)
        
        node_out = self.node_mlp(h)
        
        # Split node outputs into means and sigmas
        v_mean = node_out[:, :3]
        v_sigma = self.softplus(node_out[:, 3:6]) + 1e-6 # eps for numerical stability
        
        # Measurement Residual Bypass
        # If the sensor is available (mask=1), rely heavily on the measurement and bounded noise.
        # If missing (mask=0), rely entirely on the GNN's message passing and estimated epistemic uncertainty.
        v_mask = x[:, 3].unsqueeze(1)
        v_meas = x[:, :3]
        
        v_mean = v_mask * v_meas + (1 - v_mask) * v_mean
        # Base measurement noise is 1% (0.01)
        v_sigma = v_mask * 0.01 + (1 - v_mask) * v_sigma
        
        theta_mean = node_out[:, 6:9]
        theta_sigma = self.softplus(node_out[:, 9:12]) + 1e-6
        
        # Split edge outputs into means and sigmas
        u, v = edge_index
        edge_input = torch.cat([h[u], h[v], edge_attr], dim=-1)
        edge_out = self.edge_mlp(edge_input)
        
        i_mean = edge_out[:, :3]
        i_sigma = self.softplus(edge_out[:, 3:6]) + 1e-6
        
        return {
            "v_mean": v_mean, "v_sigma": v_sigma,
            "theta_mean": theta_mean, "theta_sigma": theta_sigma,
            "i_mean": i_mean, "i_sigma": i_sigma
        }

    def mc_dropout_predict(self, x, edge_index, edge_attr, num_samples=10):
        """
        Executes Monte Carlo Dropout to decompose total uncertainty.
        Returns the mean prediction, aleatoric sigma, and epistemic sigma.
        """
        # Ensure dropout is active
        self.train()
        
        v_means, v_sigmas = [], []
        with torch.no_grad():
            for _ in range(num_samples):
                out = self.forward(x, edge_index, edge_attr)
                v_means.append(out["v_mean"])
                v_sigmas.append(out["v_sigma"])
                
        # Stack samples: [num_samples, num_nodes, 3]
        v_means_stack = torch.stack(v_means)
        v_sigmas_stack = torch.stack(v_sigmas)
        
        # Final predictive mean
        expected_v = v_means_stack.mean(dim=0)
        
        # Epistemic Uncertainty = Variance of the predicted means
        epistemic_var = v_means_stack.var(dim=0)
        
        # Aleatoric Uncertainty = Mean of the predicted variances
        aleatoric_var = (v_sigmas_stack ** 2).mean(dim=0)
        
        self.eval() # Restore eval mode
        
        return {
            "v_mean": expected_v,
            "v_sigma_aleatoric": torch.sqrt(aleatoric_var),
            "v_sigma_epistemic": torch.sqrt(epistemic_var)
        }

class UncertaintyLoss(nn.Module):
    def __init__(self, lambda_kcl=0.1, lambda_power=0.1, lambda_voltage=0.1, base_voltage=7309.24):
        super().__init__()
        self.lambda_kcl = lambda_kcl
        self.lambda_power = lambda_power
        self.lambda_voltage = lambda_voltage
        self.base_voltage = base_voltage

    def gaussian_nll(self, mean, sigma, target):
        """Computes the Negative Log-Likelihood of a Gaussian distribution."""
        variance = sigma ** 2
        loss = 0.5 * torch.log(variance) + 0.5 * ((target - mean) ** 2) / variance
        return torch.mean(loss)

    def forward(self, preds, v_true, theta_true, i_true, edge_index, edge_attr):
        # 1. Data Loss (Gaussian NLL instead of pure MSE)
        nll_v = self.gaussian_nll(preds["v_mean"], preds["v_sigma"], v_true)
        nll_theta = self.gaussian_nll(preds["theta_mean"], preds["theta_sigma"], theta_true)
        nll_i = self.gaussian_nll(preds["i_mean"], preds["i_sigma"], i_true)
        
        l_data = nll_v + nll_theta + nll_i
        
        # 2. Physics losses calculated against the expected mean prediction
        i_mean = preds["i_mean"]
        v_mean = preds["v_mean"]
        
        # KCL
        u, v = edge_index
        num_nodes = v_mean.size(0)
        
        i_in = torch.zeros((num_nodes, 3), device=v_mean.device)
        i_in.scatter_add_(0, v.unsqueeze(1).expand(-1, 3), i_mean)
        
        i_out = torch.zeros((num_nodes, 3), device=v_mean.device)
        i_out.scatter_add_(0, u.unsqueeze(1).expand(-1, 3), i_mean)
        
        l_kcl = torch.mean((i_in - i_out) ** 2)
        
        # Voltage constraints (predictions are already natively in Per-Unit)
        v_pu = v_mean
        v_under = torch.nn.functional.relu(0.95 - v_pu)
        v_over = torch.nn.functional.relu(v_pu - 1.05)
        l_voltage = torch.mean(v_under ** 2 + v_over ** 2)
        
        # Power balance
        r_val = edge_attr[:, 0].unsqueeze(1)
        p_loss_hat = torch.sum((i_mean ** 2) * r_val)
        true_net_power = torch.sum((i_true ** 2) * r_val)
        l_power = ((true_net_power - p_loss_hat) ** 2) / (true_net_power ** 2 + 1e-8)
        
        total_loss = l_data + self.lambda_kcl * l_kcl + self.lambda_power * l_power + self.lambda_voltage * l_voltage
        
        return total_loss
