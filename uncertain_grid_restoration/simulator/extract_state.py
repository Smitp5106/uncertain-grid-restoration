import numpy as np
from opendssdirect import dss

def extract_ground_truth():
    """
    Extracts the ground truth state of the grid from OpenDSS.
    Returns a dictionary containing voltages, angles, currents, and powers.
    """
    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS solution has not converged. Cannot extract valid state.")

    buses = dss.Circuit.AllBusNames()
    voltages = {}
    angles = {}

    for bus_name in buses:
        dss.Circuit.SetActiveBus(bus_name)
        # puVmagAngle returns [pu_mag1, ang1, pu_mag2, ang2, ...]
        pu_v_mag_angle = dss.Bus.puVmagAngle()
        mags = pu_v_mag_angle[0::2]
        angs = pu_v_mag_angle[1::2]
        voltages[bus_name] = mags
        angles[bus_name] = angs

    lines = dss.Lines.AllNames()
    currents = {}
    powers = {}

    for line_name in lines:
        dss.Circuit.SetActiveElement(f"Line.{line_name}")
        
        i_mag_ang = dss.CktElement.CurrentsMagAng()
        # i_mag_ang has [mag, ang] for all conductors at all terminals
        
        pq = dss.CktElement.Powers()
        # pq has [P, Q] for all conductors at all terminals
        
        num_phases = dss.CktElement.NumPhases()
        
        # Extract terminal 1 (sending end)
        i_mags = i_mag_ang[0: 2*num_phases :2]
        p_term1 = pq[0: 2*num_phases :2]
        q_term1 = pq[1: 2*num_phases :2]
        
        currents[line_name] = i_mags
        powers[line_name] = {"P": p_term1, "Q": q_term1}

    return {
        "voltage": voltages,
        "angle": angles,
        "current": currents,
        "power": powers
    }

if __name__ == "__main__":
    import os
    # Change working directory so relative path works
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    dss("Clear")
    dss('Redirect "dss/ieee33/Master.dss"')
    dss.Solution.Solve()
    
    true_state = extract_ground_truth()
    
    print("Extracted True State.")
    print("Voltage at Bus 2:", true_state["voltage"]["2"])
    print("Angle at Bus 2:", true_state["angle"]["2"])
    
    # line names are typically lowercased in OpenDSS direct by default, so we pick the first one
    first_line = list(true_state["current"].keys())[0]
    print(f"Current at Line {first_line}:", true_state["current"][first_line])
    print(f"Power at Line {first_line}:", true_state["power"][first_line])
