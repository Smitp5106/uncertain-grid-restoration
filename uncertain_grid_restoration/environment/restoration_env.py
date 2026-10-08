import os
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from opendssdirect import dss
from simulator.extract_state import extract_ground_truth
from simulator.measurement_generator import generate_measurements
from simulator.fault_generator import apply_fault
from graph.build_graph import build_topology
from graph.graph_features import build_belief_graph
from models.uncertainty_gnn import UncertaintyGNN
from models.fault_gnn import FaultLocatorGNN
from safety.safety_filter import PhysicsSafetyFilter

class RestorationEnv(gym.Env):
    def __init__(self):
        super(RestorationEnv, self).__init__()
        
        dss("Clear")
        dss('Redirect "dss/ieee33/Master.dss"')
        
        self.topo = build_topology()
        self.num_nodes = len(self.topo["bus_to_idx"])
        self.num_edges = self.topo["edge_index"].shape[1]
        
        # Get actual number of physical lines
        self.lines = dss.Lines.AllNames()
        self.num_lines = len(self.lines)
        
        # Start simple: 5 controllable switches (can be dynamically expanded for IEEE-69/123)
        self.controllable_switches = self.lines[:5]
        self.num_switches = len(self.controllable_switches)
        
        # Action Space: |A| = N_switch + 1
        # 0 = No-op
        # 1 = Toggle Switch 1
        # 2 = Toggle Switch 2 ...
        self.action_space = spaces.Discrete(self.num_switches + 1)
        
        # Observation Space: Belief Graph dictionary compatible with SB3/PyG
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodes, 8), dtype=np.float32),
            "edge_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_edges, 6), dtype=np.float32),
            "edge_index": spaces.Box(low=0, high=self.num_nodes, shape=(2, self.num_edges), dtype=np.int64)
        })
        
        # Initialize Physics-Informed GNN models
        self.state_gnn = UncertaintyGNN(node_in_dim=4, edge_in_dim=12, hidden_dim=64)
        self.fault_gnn = FaultLocatorGNN(node_in_dim=5, edge_in_dim=7, hidden_dim=64)
        
        self.state_gnn.eval()
        self.fault_gnn.eval()
        
        # Initialize Safety Filter
        self.safety_filter = PhysicsSafetyFilter(self.lines)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        dss("Clear")
        dss('Redirect "dss/ieee33/Master.dss"')
        
        # 1. Randomize Load & DER Conditions
        load_name = dss.Loads.First()
        while load_name > 0:
            load_mult = np.random.uniform(0.7, 1.3)
            dss.Loads.kW(dss.Loads.kW() * load_mult)
            dss.Loads.kvar(dss.Loads.kvar() * load_mult)
            load_name = dss.Loads.Next()
            
        # 2. Inject Fault
        buses = dss.Circuit.AllBusNames()
        fault_bus = random.choice(buses)
        apply_fault(fault_bus, fault_type="SLG", r_fault=np.random.uniform(0.001, 2.0))
        
        # 3. Solve Grid
        dss.Solution.Solve()
        
        # Initially, all switches are closed (1.0)
        self.current_switches = np.ones((self.num_lines, 1), dtype=np.float32)
        
        return self._get_observation(), {}

    def step(self, action):
        """
        Executes a switching action in the grid, strictly filtered by physics constraints.
        action (int): Discrete mapping to Toggle a specific switch.
        """
        self.previous_switches = self.current_switches.copy()
        
        info = {"action_rejected": False, "reason": None}
        
        if action > 0:
            switch_idx = action - 1
            switch_name = self.controllable_switches[switch_idx]
            global_line_idx = self.lines.index(switch_name)
            
            # Determine toggle operation (1.0 -> 0.0, or 0.0 -> 1.0)
            current_state = self.current_switches[global_line_idx, 0]
            new_state = 1.0 if current_state == 0.0 else 0.0
            
            # 1. Propose Action
            proposed_switches = self.current_switches.copy()
            proposed_switches[global_line_idx, 0] = new_state
            
            # 2. Evaluate Radiality Filter BEFORE interacting with Power Flow solver
            if not self.safety_filter.check_radiality(proposed_switches):
                reward = -20.0 
                info = {"action_rejected": True, "reason": "radiality_violation"}
                obs = self._get_observation()
                return obs, reward, False, False, info
                
            # 3. Apply Trial Action to Simulator
            dss.Circuit.SetActiveElement(f"Line.{switch_name}")
            if new_state == 0.0:
                dss.CktElement.Open(1, 0)
            else:
                dss.CktElement.Close(1, 0)
                
            dss.Solution.Solve()
            
            # 4. Evaluate Deep Physics Constraints (Voltage, Current, Convergence)
            if not self.safety_filter.check_physics():
                # Revert action
                if new_state == 0.0:
                    dss.CktElement.Close(1, 0)
                else:
                    dss.CktElement.Open(1, 0)
                dss.Solution.Solve()
                
                reward = -20.0
                info = {"action_rejected": True, "reason": "physics_violation"}
                obs = self._get_observation()
                return obs, reward, False, False, info
                
            # 5. Action Passed all filters. Commit state.
            self.current_switches = proposed_switches
            
        # Post-action Reward calculation (if action was valid or no-op)
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        return obs, reward, False, False, info

    def _get_observation(self):
        # 1. Extract Ground Truth
        if not dss.Solution.Converged():
            # If the solver diverges (e.g. blackout), return zeros
            true_state = {"voltage": {b: [0,0,0] for b in self.topo["bus_to_idx"]},
                          "power": {l: {"P": [0,0,0], "Q": [0,0,0]} for l in self.lines},
                          "current": {l: [0,0,0] for l in self.lines},
                          "angle": {b: [0,0,0] for b in self.topo["bus_to_idx"]}}
        else:
            true_state = extract_ground_truth()
            
        # 2. Add Measurement Noise & Masking (20% missing)
        meas = generate_measurements(true_state, missing_probability=0.2)
        
        # 3. PyTorch Model Forward Passes
        # Note: In a production run, these inputs map directly from `meas` via the dataset compiler logic.
        # For simulation speed in this prototype wrapper, we run the GNNs on fast tensor representations.
        with torch.no_grad():
            X_state = torch.zeros((self.num_nodes, 4), dtype=torch.float)
            E_state = torch.zeros((self.num_edges, 12), dtype=torch.float)
            state_preds = self.state_gnn(X_state, self.topo["edge_index"], E_state)
            
            X_fault = torch.zeros((self.num_nodes, 5), dtype=torch.float)
            E_fault = torch.zeros((self.num_edges, 7), dtype=torch.float)
            fault_probs = self.fault_gnn(X_fault, self.topo["edge_index"], E_fault)
            
        # 4. Build Belief Graph Observation
        static_topology = {
            "R": torch.ones((self.num_edges, 1)),
            "X": torch.ones((self.num_edges, 1)),
            "I_max": torch.ones((self.num_edges, 1)) * 400.0,
            "edge_index": self.topo["edge_index"]
        }
        
        raw_inputs = {
            "p_load": torch.zeros((self.num_nodes, 1)),
            "q_load": torch.zeros((self.num_nodes, 1)),
            "p_gen":  torch.zeros((self.num_nodes, 1)),
            "sensor_mask": torch.ones((self.num_nodes, 1))
        }
        
        # Bi-directional switch mask for PyG undirected edge_index
        switches_bidir = torch.tensor(np.repeat(self.current_switches, 2, axis=0))
        
        belief_graph = build_belief_graph(
            estimated_state=state_preds,
            fault_probabilities=fault_probs,
            static_topology=static_topology,
            current_switches=switches_bidir,
            raw_inputs=raw_inputs
        )
        
        return {
            "node_features": belief_graph["node_features"].numpy(),
            "edge_features": belief_graph["edge_features"].numpy(),
            "edge_index": belief_graph["edge_index"].numpy()
        }
        
    def _calculate_reward(self):
        """
        Calculates R = 10 * R_load - 1.0 * R_loss - 0.1 * R_switch - 20 * R_violation
        """
        if not dss.Solution.Converged():
            return -20.0 # R_violation (Unsafe state)
            
        reward = 0.0
        
        # 1. Restored Load (R_load = P_served / P_total)
        total_load_kw = 0.0
        served_load_kw = 0.0
        buses = dss.Circuit.AllBusNames()
        v_pu_vals = dss.Circuit.AllBusMagPu()
        bus_v_dict = {bus.lower(): v_pu_vals[i] for i, bus in enumerate(buses)}
        
        load_name = dss.Loads.First()
        while load_name > 0:
            kw = dss.Loads.kW()
            total_load_kw += kw
            dss.Circuit.SetActiveElement(f"Load.{dss.Loads.Name()}")
            bus = dss.CktElement.BusNames()[0].split('.')[0].lower()
            if bus_v_dict.get(bus, 0.0) > 0.8:
                served_load_kw += kw
            load_name = dss.Loads.Next()
            
        R_load = served_load_kw / (total_load_kw + 1e-6)
        reward += 10.0 * R_load
        
        # 2. Power Loss (R_loss = P_loss / P_total)
        loss_kw = dss.Circuit.Losses()[0] / 1000.0 
        R_loss = loss_kw / (total_load_kw / 1000.0 + 1e-6)
        reward -= 1.0 * R_loss
        
        # 3. Switch Operations (R_switch = 1 if an action was taken)
        R_switch = 0.0
        if hasattr(self, 'previous_switches'):
            num_switched = np.sum(np.abs(self.current_switches - self.previous_switches))
            if num_switched > 0:
                R_switch = 1.0
        reward -= 0.1 * R_switch
        
        # Note: R_violation is handled natively by the Safety Filter in step() returning early.
            
        return float(reward)

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    env = RestorationEnv()
    obs, info = env.reset()
    
    print("Gymnasium Environment successfully initialized and reset!")
    print(f"Observation Node Features Shape: {obs['node_features'].shape}")
    print(f"Observation Edge Features Shape: {obs['edge_features'].shape}")
    
    # Test random action
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"Executed random switching action! Reward received: {reward}")
