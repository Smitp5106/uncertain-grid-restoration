import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear

class BaselineGNN(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64):
        super(BaselineGNN, self).__init__()
        
        # Input Node Embedding
        self.node_emb = Linear(node_in_dim, hidden_dim)
        
        # 3 GNN layers capable of incorporating edge attributes
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        
        # Node-level MLP (Estimates V_hat, theta_hat)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6) # 3 phases for V, 3 phases for theta
        )
        
        # Edge-level MLP (Estimates I_hat)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # 3 phases for I
        )

    def forward(self, x, edge_index, edge_attr):
        # Project node features
        h = F.relu(self.node_emb(x))
        
        # Message passing layers
        h = F.relu(self.conv1(h, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        h = self.conv3(h, edge_index, edge_attr) # No ReLU on final embedding
        
        # Output V and Theta
        node_out = self.node_mlp(h)
        v_hat = node_out[:, :3]
        theta_hat = node_out[:, 3:]
        
        # Output I (combine source node, target node, and edge features)
        u, v = edge_index
        edge_input = torch.cat([h[u], h[v], edge_attr], dim=-1)
        i_hat = self.edge_mlp(edge_input)
        
        return v_hat, theta_hat, i_hat
