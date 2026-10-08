import torch
import numpy as np
from opendssdirect import dss

def build_topology():
    """
    Builds the PyTorch Geometric graph topology from the OpenDSS model.
    Extracts nodes (buses) and edges (lines).
    Returns edge_index, edge_attr, and a mapping from bus name to node index.
    """
    if len(dss.Circuit.AllBusNames()) == 0:
        raise RuntimeError("No circuit loaded in OpenDSS.")

    buses = dss.Circuit.AllBusNames()
    num_nodes = len(buses)
    
    # Create mapping from bus name to integer index
    bus_to_idx = {bus.lower(): i for i, bus in enumerate(buses)}
    
    sources = []
    targets = []
    edge_attrs = []
    
    # Iterate through all lines
    for line_name in dss.Lines.AllNames():
        dss.Lines.Name(line_name)
        
        bus1_full = dss.Lines.Bus1()
        bus2_full = dss.Lines.Bus2()
        
        # OpenDSS bus names can include node/phase numbers (e.g., "bus1.1.2.3")
        bus1 = bus1_full.split('.')[0].lower()
        bus2 = bus2_full.split('.')[0].lower()
        
        if bus1 not in bus_to_idx or bus2 not in bus_to_idx:
            continue
            
        u = bus_to_idx[bus1]
        v = bus_to_idx[bus2]
        
        # Line features: R1, X1 (Positive sequence resistance and reactance)
        r1 = dss.Lines.R1()
        x1 = dss.Lines.X1()
        length = dss.Lines.Length()
        
        # Total R and X for the line
        R = r1 * length
        X = x1 * length
        
        # Add directed edges for PyTorch Geometric (undirected representation)
        sources.extend([u, v])
        targets.extend([v, u])
        
        # Same features for both directions [R, X]
        edge_attrs.extend([[R, X], [R, X]])

    # Convert to PyTorch tensors
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # Initial placeholder for node features (e.g., P, Q measurements)
    # We will likely inject active measurements into these later.
    node_features = torch.zeros((num_nodes, 2), dtype=torch.float)
    
    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "node_features": node_features,
        "bus_to_idx": bus_to_idx,
        "idx_to_bus": {i: b for b, i in bus_to_idx.items()}
    }

if __name__ == "__main__":
    import os
    # Change working directory so relative path works
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    dss("Clear")
    dss('Redirect "dss/ieee33/Master.dss"')
    dss.Solution.Solve()
    
    graph_data = build_topology()
    
    print("Graph topology built successfully!")
    print(f"Number of nodes (buses): {len(graph_data['bus_to_idx'])}")
    print(f"Edge Index shape: {graph_data['edge_index'].shape}")
    print(f"Edge Attr shape: {graph_data['edge_attr'].shape}")
    print(f"Node Features shape: {graph_data['node_features'].shape}")
    
    print("\nFirst 10 edges in edge_index (undirected pairs):")
    print(graph_data['edge_index'][:, :10])
