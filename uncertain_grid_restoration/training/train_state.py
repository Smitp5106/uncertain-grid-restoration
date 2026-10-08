import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to sys path
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

from models.gnn_baseline import BaselineGNN

def collate_fn(batch):
    """Batches disjoint graphs into a single large graph structure for PyG."""
    node_features = []
    edge_index = []
    edge_features = []
    v_true = []
    theta_true = []
    i_true = []
    
    node_offset = 0
    for sample in batch:
        num_nodes = sample["node_features"].shape[0]
        
        node_features.append(sample["node_features"])
        edge_features.append(sample["edge_features"])
        
        # Shift edge indices by the number of nodes already accumulated
        edge_index.append(sample["edge_index"] + node_offset)
        
        v_true.append(sample["V_true"])
        theta_true.append(sample["theta_true"])
        i_true.append(sample["I_true"])
        
        node_offset += num_nodes
        
    return {
        "node_features": torch.cat(node_features, dim=0),
        "edge_index": torch.cat(edge_index, dim=1),
        "edge_features": torch.cat(edge_features, dim=0),
        "V_true": torch.cat(v_true, dim=0),
        "theta_true": torch.cat(theta_true, dim=0),
        "I_true": torch.cat(i_true, dim=0)
    }

def train():
    dataset_path = "data/state_estimation/dataset_1k.pt"
    print(f"Loading {dataset_path}...")
    dataset = torch.load(dataset_path)
    
    # Train / Val Split
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Initialize Model
    # Node features: 4 [V1, V2, V3, mask]
    # Edge features: 12 [R, X, P123, Q123, I123, mask]
    model = BaselineGNN(node_in_dim=4, edge_in_dim=12, hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # The loss function as specified: L = MSE(V) + MSE(theta) + MSE(I)
    mse = nn.MSELoss()
    
    print("\nStarting Training (10 Epochs):")
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            x = batch["node_features"]
            edge_index = batch["edge_index"]
            edge_attr = batch["edge_features"]
            
            v_hat, theta_hat, i_hat = model(x, edge_index, edge_attr)
            
            loss_v = mse(v_hat, batch["V_true"])
            loss_theta = mse(theta_hat, batch["theta_true"])
            loss_i = mse(i_hat, batch["I_true"])
            
            loss = loss_v + loss_theta + loss_i
            loss.backward()
            
            # Gradient clipping is helpful for large unscaled values
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                v_hat, theta_hat, i_hat = model(batch["node_features"], batch["edge_index"], batch["edge_features"])
                loss = mse(v_hat, batch["V_true"]) + mse(theta_hat, batch["theta_true"]) + mse(i_hat, batch["I_true"])
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.2f} | Val Loss: {avg_val_loss:.2f}")

if __name__ == "__main__":
    train()
