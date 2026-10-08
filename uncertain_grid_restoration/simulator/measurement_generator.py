import numpy as np

def generate_measurements(true_state, missing_probability=0.2, v_std=0.01, p_std=0.02, q_std=0.02, i_std=0.02):
    """
    Generates noisy measurements and an explicit availability mask.
    Missing measurements are masked (0 value + 0 mask) to maintain tensor shape for GNNs.
    
    Args:
        true_state: Dictionary containing true voltage, angle, current, power.
        missing_probability: Probability that a sensor is missing (0.0 to 1.0).
        v_std: Standard deviation of voltage noise (fraction of true magnitude).
        p_std: Standard deviation for active power noise (fraction of true value).
        q_std: Standard deviation for reactive power noise (fraction of true value).
        i_std: Standard deviation for current noise (fraction of true value).
        
    Returns:
        measurements dict with 'values' and 'mask' for each type.
    """
    buses = list(true_state["voltage"].keys())
    lines = list(true_state["power"].keys())
    
    num_buses = len(buses)
    num_lines = len(lines)
    
    # Generate boolean masks (1 = available, 0 = missing)
    bus_mask = (np.random.rand(num_buses) > missing_probability).astype(int)
    # Ensure source bus (index 0) always has a voltage sensor
    bus_mask[0] = 1 
    
    line_mask = (np.random.rand(num_lines) > missing_probability).astype(int)
    
    measurements = {
        "voltage": {"values": {}, "mask": {}},
        "power": {"values": {}, "mask": {}},
        "current": {"values": {}, "mask": {}}
    }
    
    # 1. Voltage Measurements
    for idx, bus in enumerate(buses):
        is_available = bus_mask[idx]
        measurements["voltage"]["mask"][bus] = is_available
        
        noisy_v = []
        for v in true_state["voltage"][bus]:
            if is_available:
                noise = np.random.normal(0, abs(v) * v_std if v != 0 else v_std)
                noisy_v.append(v + noise)
            else:
                noisy_v.append(0.0) # Missing encoded as 0 with mask=0
                
        measurements["voltage"]["values"][bus] = noisy_v
        
    # 2. Line Power and Current Measurements
    for idx, line in enumerate(lines):
        is_available = line_mask[idx]
        measurements["power"]["mask"][line] = is_available
        measurements["current"]["mask"][line] = is_available
        
        noisy_p = []
        noisy_q = []
        noisy_i = []
        
        for p, q in zip(true_state["power"][line]["P"], true_state["power"][line]["Q"]):
            if is_available:
                noise_p = np.random.normal(0, abs(p) * p_std if p != 0 else p_std)
                noise_q = np.random.normal(0, abs(q) * q_std if q != 0 else q_std)
                noisy_p.append(p + noise_p)
                noisy_q.append(q + noise_q)
            else:
                noisy_p.append(0.0)
                noisy_q.append(0.0)
                
        for i_mag in true_state["current"][line]:
            if is_available:
                noise_i = np.random.normal(0, abs(i_mag) * i_std if i_mag != 0 else i_std)
                noisy_i.append(i_mag + noise_i)
            else:
                noisy_i.append(0.0)
                
        measurements["power"]["values"][line] = {"P": noisy_p, "Q": noisy_q}
        measurements["current"]["values"][line] = noisy_i
        
    return measurements

if __name__ == "__main__":
    import pickle
    import os
    
    # Change working directory so relative path works
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    with open("data/state_estimation/scenarios_1k.pkl", "rb") as f:
        scenarios = pickle.load(f)
        
    sample_true_state = scenarios[0]
    
    # 20% missing probability
    noisy_state = generate_measurements(sample_true_state, missing_probability=0.2)
    
    print("Measurement masking test:\n")
    print(f"{'Bus':<10} {'True V (Ph 1)':<20} {'Meas V (Ph 1)':<20} {'Mask'}")
    print("-" * 60)
    
    buses = list(sample_true_state["voltage"].keys())
    # Display first 10 buses
    for bus in buses[:10]:
        true_v = round(sample_true_state["voltage"][bus][0], 2)
        meas_v = round(noisy_state["voltage"]["values"][bus][0], 2)
        mask_val = noisy_state["voltage"]["mask"][bus]
        
        meas_str = str(meas_v) if mask_val == 1 else "MISSING (0.0)"
        print(f"{bus:<10} {true_v:<20} {meas_str:<20} {mask_val}")
