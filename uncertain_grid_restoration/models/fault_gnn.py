import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class FaultDetectorGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64):
        """
        GNN to classify a scenario as Normal (0) or Faulty (1).
        
        node_in_dim corresponds to:
            [estimated_V, estimated_theta, measurement_residual_V, physics_residual_KCL, sensor_mask]
            
        edge_in_dim corresponds to:
            [estimated_I, measurement_residual_I, measurement_residual_P, measurement_residual_Q, R, X, sensor_mask]
        """
        super(FaultDetectorGNN, self).__init__()
        
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        
        # Message passing layers
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        
        # Graph-level Readout MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # Outputs P(fault) -> [0, 1]
        )
        
    def forward(self, x, edge_index, edge_attr, batch_idx):
        """
        x: Node features (estimated state + residuals)
        edge_index: Graph topology
        edge_attr: Edge features (estimated state + residuals)
        batch_idx: Indicates which graph in the mini-batch each node belongs to
        """
        # Node embedding
        h = F.relu(self.node_emb(x))
        
        # Graph convolutions
        h = F.relu(self.conv1(h, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        h = F.relu(self.conv3(h, edge_index, edge_attr))
        
        # Pool node embeddings into a single graph embedding
        h_graph = global_mean_pool(h, batch_idx)
        
        # Predict P(fault)
        p_fault = self.mlp(h_graph)
        
        return p_fault

class FaultLocatorGNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64):
        """
        GNN to predict the exact line where a fault occurred.
        Outputs a probability array [num_lines, 1].
        """
        super(FaultLocatorGNN, self).__init__()
        
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_in_dim, add_self_loops=False)
        
        # Edge-level Readout MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Outputs P(fault) for each specific line
        )
        
    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.node_emb(x))
        
        h = F.relu(self.conv1(h, edge_index, edge_attr))
        h = F.relu(self.conv2(h, edge_index, edge_attr))
        h = F.relu(self.conv3(h, edge_index, edge_attr))
        
        # To predict edge properties, concatenate the embeddings of the connected nodes (u and v) 
        # alongside the edge attributes.
        u, v = edge_index
        edge_input = torch.cat([h[u], h[v], edge_attr], dim=-1)
        
        # Predict P(fault_line_i)
        p_fault_line = self.edge_mlp(edge_input)
        
        return p_fault_line

def compute_residuals(v_meas, v_mask, v_hat, i_meas, i_mask, i_hat, edge_index):
    """
    Helper function to dynamically calculate the physics and measurement residuals.
    These residuals act as the primary signal that a fault has occurred.
    """
    # 1. Measurement Residuals (only computed where a sensor actually exists)
    v_res = v_mask * (v_meas - v_hat)
    i_res = i_mask * (i_meas - i_hat)
    
    # 2. Physics Residuals (Kirchhoff's Current Law Mismatch)
    u, v = edge_index
    num_nodes = v_hat.size(0)
    
    i_in = torch.zeros((num_nodes, 3), device=v_hat.device)
    i_in.scatter_add_(0, v.unsqueeze(1).expand(-1, 3), i_hat)
    
    i_out = torch.zeros((num_nodes, 3), device=v_hat.device)
    i_out.scatter_add_(0, u.unsqueeze(1).expand(-1, 3), i_hat)
    
    # Under normal operation, kcl_res should be ~0. 
    # During a fault, current heavily leaks to ground/other phases, causing a massive KCL spike.
    kcl_res = i_in - i_out 
    
    return v_res, i_res, kcl_res
