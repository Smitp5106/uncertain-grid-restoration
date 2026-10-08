import torch

def build_belief_graph(
    estimated_state,       # Dict from UncertaintyGNN (v_mean, v_sigma, theta_mean, theta_sigma, i_mean)
    fault_probabilities,   # Tensor [num_edges, 1] from FaultLocatorGNN
    static_topology,       # Dict with R, X, I_max, edge_index
    current_switches,      # Tensor [num_edges, 1] (1=closed, 0=open)
    raw_inputs             # Dict with P_load, Q_load, P_gen, sensor_mask
):
    """
    Constructs the Belief Graph to act as the observation state for the RL Agent.
    This encodes the probabilistic uncertainty of both the physical state and the fault location.
    
    Node Feature Vector (dim=8):
    [V_hat, theta_hat, sigma_V, sigma_theta, P_load, Q_load, P_generation, sensor_mask]
    
    Edge Feature Vector (dim=6):
    [R, X, I_hat, I_max, switch_status, fault_probability]
    """
    
    # --- 1. Construct Node Features ---
    # For a 3-phase system, we can average across phases for a simplified RL action space, 
    # or flatten it. We will average here to match the 1D conceptual request.
    v_hat = estimated_state["v_mean"].mean(dim=1, keepdim=True)
    theta_hat = estimated_state["theta_mean"].mean(dim=1, keepdim=True)
    sigma_v = estimated_state["v_sigma"].mean(dim=1, keepdim=True)
    sigma_theta = estimated_state["theta_sigma"].mean(dim=1, keepdim=True)
    
    p_load = raw_inputs["p_load"]
    q_load = raw_inputs["q_load"]
    p_gen = raw_inputs["p_gen"]
    sensor_mask = raw_inputs["sensor_mask"]
    
    # [num_nodes, 8]
    node_features = torch.cat([
        v_hat, 
        theta_hat, 
        sigma_v, 
        sigma_theta, 
        p_load, 
        q_load, 
        p_gen, 
        sensor_mask
    ], dim=-1)
    
    # --- 2. Construct Edge Features ---
    r = static_topology["R"]
    x = static_topology["X"]
    i_max = static_topology["I_max"]
    
    i_hat = estimated_state["i_mean"].mean(dim=1, keepdim=True)
    
    # [num_edges, 6]
    edge_features = torch.cat([
        r, 
        x, 
        i_hat, 
        i_max, 
        current_switches, 
        fault_probabilities
    ], dim=-1)
    
    return {
        "node_features": node_features,
        "edge_index": static_topology["edge_index"],
        "edge_features": edge_features
    }

if __name__ == "__main__":
    # Dummy test to verify the tensor shapes map perfectly
    num_nodes = 33
    num_edges = 64
    
    dummy_estimated = {
        "v_mean": torch.rand(num_nodes, 3),
        "v_sigma": torch.rand(num_nodes, 3),
        "theta_mean": torch.rand(num_nodes, 3),
        "theta_sigma": torch.rand(num_nodes, 3),
        "i_mean": torch.rand(num_edges, 3)
    }
    
    dummy_fault_probs = torch.rand(num_edges, 1) # P(line_i)
    
    dummy_topology = {
        "R": torch.rand(num_edges, 1),
        "X": torch.rand(num_edges, 1),
        "I_max": torch.ones(num_edges, 1) * 400.0,
        "edge_index": torch.randint(0, num_nodes, (2, num_edges))
    }
    
    dummy_switches = torch.ones(num_edges, 1)
    
    dummy_inputs = {
        "p_load": torch.rand(num_nodes, 1),
        "q_load": torch.rand(num_nodes, 1),
        "p_gen": torch.zeros(num_nodes, 1),
        "sensor_mask": torch.randint(0, 2, (num_nodes, 1)).float()
    }
    
    belief_graph = build_belief_graph(
        estimated_state=dummy_estimated,
        fault_probabilities=dummy_fault_probs,
        static_topology=dummy_topology,
        current_switches=dummy_switches,
        raw_inputs=dummy_inputs
    )
    
    print("Belief Graph successfully built!")
    print(f"Node Features Shape: {belief_graph['node_features'].shape} (Expected: [33, 8])")
    print(f"Edge Features Shape: {belief_graph['edge_features'].shape} (Expected: [64, 6])")
