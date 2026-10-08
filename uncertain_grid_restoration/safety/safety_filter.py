import numpy as np
from opendssdirect import dss

class PhysicsSafetyFilter:
    def __init__(self, lines):
        """
        The Safety Filter prevents the RL agent from committing physically 
        destructive actions to the real (or simulated) distribution grid.
        """
        self.lines = lines
        
    def check_radiality(self, proposed_switches):
        """
        Rigorous topological check to prevent loop formation and ensure connectivity.
        Evaluated BEFORE running the power flow solver.
        """
        import networkx as nx
        
        G = nx.Graph()
        # Add all buses to ensure isolated nodes are tracked
        buses = dss.Circuit.AllBusNames()
        G.add_nodes_from([b.lower() for b in buses])
        
        # Add active edges
        for idx, line in enumerate(self.lines):
            if proposed_switches[idx, 0] == 1.0:
                dss.Circuit.SetActiveElement(f"Line.{line}")
                bus1 = dss.Lines.Bus1().split('.')[0].lower()
                bus2 = dss.Lines.Bus2().split('.')[0].lower()
                G.add_edge(bus1, bus2)
                
        # 1. Cycle Detection (Must be acyclic)
        try:
            nx.find_cycle(G)
            is_acyclic = False
        except nx.NetworkXNoCycle:
            is_acyclic = True
            
        # 2. Connectivity Check (Must be fully connected)
        # Note: If strict make-before-break switching is not used, 
        # isolating a fault will temporarily disconnect the graph.
        # However, to mathematically guarantee a single spanning tree:
        is_connected = nx.is_connected(G)
        
        # Enforce exact mathematical definition of a radial connected grid
        # Note: In a single-switch toggle action space, enforcing connectivity 
        # means the agent can never open a switch without causing a rejection, 
        # unless it is performing a multi-switch 'make-before-break' sequence.
        return is_connected and is_acyclic
        
    def check_physics(self):
        """
        Deep physical bounds check. 
        Evaluated AFTER a trial power flow is solved.
        """
        # 1. Convergence Check (e.g. closing directly into a hard short circuit)
        if not dss.Solution.Converged():
            return False
            
        # 2. Voltage Safety Constraints
        v_pu_vals = dss.Circuit.AllBusMagPu()
        for v in v_pu_vals:
            # Only check energized buses. 
            # Dead buses (v < 0.1) are fine (just unserved load, not a safety risk)
            if v > 0.1 and (v < 0.95 or v > 1.05):
                return False
                
        # 3. Thermal Overload / Current Constraints
        for line in self.lines:
            dss.Circuit.SetActiveElement(f"Line.{line}")
            currents = dss.CktElement.CurrentsMagAng()
            max_i = max(currents[0::2]) if currents else 0
            
            # If line current exceeds its thermal rating, it's unsafe
            if max_i > dss.CktElement.NormalAmps():
                return False
                
        return True
