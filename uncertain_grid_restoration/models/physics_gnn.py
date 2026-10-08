import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear

class PhysicsGNN(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64):
        super(PhysicsGNN, self).__init__()
        self.node_emb = Linear(node_in_dim, hidden_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6) # V_hat (3), theta_hat (3)
        )
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # I_hat (3)
        )

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.node_emb(x))
        h = F.relu(self.conv1(h, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        h = self.conv3(h, edge_index, edge_attr)
        
        node_out = self.node_mlp(h)
        v_hat = node_out[:, :3]
        theta_hat = node_out[:, 3:]
        
        u, v = edge_index
        edge_input = torch.cat([h[u], h[v], edge_attr], dim=-1)
        i_hat = self.edge_mlp(edge_input)
        
        return v_hat, theta_hat, i_hat

class PhysicsLoss(nn.Module):
    def __init__(self, lambda_kcl=0.1, lambda_power=0.1, lambda_voltage=0.1, base_voltage=7309.24):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_kcl = lambda_kcl
        self.lambda_power = lambda_power
        self.lambda_voltage = lambda_voltage
        self.base_voltage = base_voltage

    def forward(self, v_hat, theta_hat, i_hat, v_true, theta_true, i_true, edge_index, edge_attr):
        """
        Calculates L = L_data + λ1 L_kcl + λ2 L_power + λ3 L_voltage
        """
        # 1. Data Loss (Supervised MSE)
        l_data = self.mse(v_hat, v_true) + self.mse(theta_hat, theta_true) + self.mse(i_hat, i_true)
        
        # 2. KCL Loss (Sum I_in - Sum I_out = 0)
        u, v = edge_index
        num_nodes = v_hat.size(0)
        
        # Scatter add to sum currents at nodes
        i_in = torch.zeros((num_nodes, 3), device=v_hat.device)
        i_in.scatter_add_(0, v.unsqueeze(1).expand(-1, 3), i_hat)
        
        i_out = torch.zeros((num_nodes, 3), device=v_hat.device)
        i_out.scatter_add_(0, u.unsqueeze(1).expand(-1, 3), i_hat)
        
        kcl_mismatch = i_in - i_out
        l_kcl = torch.mean(kcl_mismatch ** 2)
        
        # 3. Voltage Constraint Loss (Penalize V < 0.95pu or V > 1.05pu)
        v_pu = v_hat
        
        v_under = torch.nn.functional.relu(0.95 - v_pu)
        v_over = torch.nn.functional.relu(v_pu - 1.05)
        l_voltage = torch.mean(v_under ** 2 + v_over ** 2)
        
        # 4. Power Balance Loss (P_gen - P_load - P_loss = 0)
        # We calculate the residual of global loss matching
        r_val = edge_attr[:, 0].unsqueeze(1) # Resistance
        
        # Predicted P_loss = Sum( I_hat^2 * R )
        p_loss_hat = torch.sum((i_hat ** 2) * r_val)
        
        # To represent P_gen - P_load inherently, we use the true net power 
        # (which inherently equals the exact true losses in the system)
        true_net_power = torch.sum((i_true ** 2) * r_val)
        
        # Power residual = (P_gen - P_load) - P_loss_hat
        power_residual = true_net_power - p_loss_hat
        
        # Normalize the MSE so it doesn't wildly dominate the gradients
        l_power = (power_residual ** 2) / (true_net_power ** 2 + 1e-8)
        
        # Total Loss
        total_loss = l_data + self.lambda_kcl * l_kcl + self.lambda_power * l_power + self.lambda_voltage * l_voltage
        
        metrics = {
            "Total": total_loss.item(),
            "Data": l_data.item(),
            "KCL": (self.lambda_kcl * l_kcl).item(),
            "Voltage": (self.lambda_voltage * l_voltage).item(),
            "Power": (self.lambda_power * l_power).item()
        }
        
        return total_loss, metrics
