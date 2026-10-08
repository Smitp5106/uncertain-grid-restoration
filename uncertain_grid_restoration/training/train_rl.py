import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATv2Conv

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

from environment.restoration_env import RestorationEnv

class GraphFlattenExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(GraphFlattenExtractor, self).__init__(observation_space, features_dim)
        node_shape = observation_space.spaces['node_features'].shape
        edge_shape = observation_space.spaces['edge_features'].shape
        flat_dim = (node_shape[0] * node_shape[1]) + (edge_shape[0] * edge_shape[1])
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 512), nn.ReLU(),
            nn.Linear(512, features_dim), nn.ReLU()
        )

    def forward(self, observations):
        batch_size = observations["node_features"].shape[0]
        node_flat = observations["node_features"].reshape(batch_size, -1)
        edge_flat = observations["edge_features"].reshape(batch_size, -1)
        x = torch.cat([node_flat, edge_flat], dim=1)
        return self.encoder(x)


class GraphGNNExtractor(BaseFeaturesExtractor):
    """
    A native Graph Neural Network Feature Extractor for PPO.
    Processes the Belief Graph using Message Passing before routing to the Actor-Critic.
    """
    def __init__(self, observation_space, features_dim=256):
        super(GraphGNNExtractor, self).__init__(observation_space, features_dim)
        
        self.num_nodes = observation_space.spaces['node_features'].shape[0]
        self.num_edges = observation_space.spaces['edge_features'].shape[0]
        
        node_dim = observation_space.spaces['node_features'].shape[1]
        edge_dim = observation_space.spaces['edge_features'].shape[1]
        
        hidden_dim = 32
        self.conv1 = GATv2Conv(node_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False)
        
        # Flatten the GNN node embeddings for the PPO MLP
        self.fc = nn.Sequential(
            nn.Linear(self.num_nodes * hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        B = observations["node_features"].shape[0]
        device = observations["node_features"].device
        
        # Construct disjoint batch for PyG
        x = observations["node_features"].reshape(B * self.num_nodes, -1)
        edge_attr = observations["edge_features"].reshape(B * self.num_edges, -1)
        edge_index = observations["edge_index"].to(torch.long)
        
        # Shift edge_index for batching
        offsets = torch.arange(B, device=device).view(B, 1, 1) * self.num_nodes
        batched_edge_index = (edge_index + offsets).reshape(2, -1)
        
        # Message Passing
        h = F.relu(self.conv1(x, batched_edge_index, edge_attr))
        h = F.relu(self.conv2(h, batched_edge_index, edge_attr))
        
        h_flat = h.reshape(B, self.num_nodes * 32)
        return self.fc(h_flat)


def compare_models():
    env = RestorationEnv()
    
    print("\n--- Training Model 1: Baseline PPO (Flatten Encoder) ---")
    kwargs_flat = dict(features_extractor_class=GraphFlattenExtractor, features_extractor_kwargs=dict(features_dim=256))
    model_flat = PPO("MultiInputPolicy", env, policy_kwargs=kwargs_flat, verbose=0)
    model_flat.learn(total_timesteps=1000)
    model_flat.save("data/processed/ppo_flatten")
    print("Baseline PPO saved.")
    
    print("\n--- Training Model 2: GNN-PPO (Native Graph Encoder) ---")
    kwargs_gnn = dict(features_extractor_class=GraphGNNExtractor, features_extractor_kwargs=dict(features_dim=256))
    model_gnn = PPO("MultiInputPolicy", env, policy_kwargs=kwargs_gnn, verbose=0)
    model_gnn.learn(total_timesteps=1000)
    model_gnn.save("data/processed/ppo_gnn")
    print("GNN-PPO saved.")
    
    print("\nComparison Complete! Both architectures successfully integrated and trained.")

if __name__ == "__main__":
    compare_models()
