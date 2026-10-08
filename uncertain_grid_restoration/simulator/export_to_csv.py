import os
import sys
import csv
import gc
import pickle
import torch
from tqdm import tqdm

def export_scenarios_pkl_to_csv(pkl_path, output_dir=None, prefix=None):
    """
    Exports a scenario .pkl file (containing voltage, angle, current, power)
    into structured tabular CSV files.
    """
    if not os.path.exists(pkl_path):
        print(f"[!] Warning: File {pkl_path} does not exist. Skipping.")
        return

    if output_dir is None:
        output_dir = os.path.dirname(pkl_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = prefix or os.path.splitext(os.path.basename(pkl_path))[0]
    buses_csv = os.path.join(output_dir, f"{base_name}_buses.csv")
    lines_csv = os.path.join(output_dir, f"{base_name}_lines.csv")
    summary_csv = os.path.join(output_dir, f"{base_name}_summary.csv")

    print(f"\n[+] Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        scenarios = pickle.load(f)

    num_scenarios = len(scenarios)
    print(f"    Loaded {num_scenarios:,} scenarios. Exporting to CSV...")

    has_fault_labels = "fault_label" in scenarios[0]

    # 1. Export Bus Measurements & Ground Truths
    print(f"    -> Writing {buses_csv}...")
    with open(buses_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_id", "bus",
            "v_phase1", "v_phase2", "v_phase3",
            "angle_phase1", "angle_phase2", "angle_phase3"
        ])
        for s_idx, sc in enumerate(tqdm(scenarios, desc="    Buses")):
            voltage_dict = sc.get("voltage", {})
            angle_dict = sc.get("angle", {})
            for bus, v_vals in voltage_dict.items():
                ang_vals = angle_dict.get(bus, [0.0, 0.0, 0.0])
                v1 = v_vals[0] if len(v_vals) > 0 else 0.0
                v2 = v_vals[1] if len(v_vals) > 1 else 0.0
                v3 = v_vals[2] if len(v_vals) > 2 else 0.0
                a1 = ang_vals[0] if len(ang_vals) > 0 else 0.0
                a2 = ang_vals[1] if len(ang_vals) > 1 else 0.0
                a3 = ang_vals[2] if len(ang_vals) > 2 else 0.0
                writer.writerow([s_idx, bus, v1, v2, v3, a1, a2, a3])

    # 2. Export Line Currents & Powers
    print(f"    -> Writing {lines_csv}...")
    with open(lines_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario_id", "line",
            "i_phase1", "i_phase2", "i_phase3",
            "p_phase1", "p_phase2", "p_phase3",
            "q_phase1", "q_phase2", "q_phase3"
        ])
        for s_idx, sc in enumerate(tqdm(scenarios, desc="    Lines")):
            current_dict = sc.get("current", {})
            power_dict = sc.get("power", {})
            for line, i_vals in current_dict.items():
                p_vals = power_dict.get(line, {}).get("P", [0.0, 0.0, 0.0])
                q_vals = power_dict.get(line, {}).get("Q", [0.0, 0.0, 0.0])
                i1 = i_vals[0] if len(i_vals) > 0 else 0.0
                i2 = i_vals[1] if len(i_vals) > 1 else 0.0
                i3 = i_vals[2] if len(i_vals) > 2 else 0.0
                p1 = p_vals[0] if len(p_vals) > 0 else 0.0
                p2 = p_vals[1] if len(p_vals) > 1 else 0.0
                p3 = p_vals[2] if len(p_vals) > 2 else 0.0
                q1 = q_vals[0] if len(q_vals) > 0 else 0.0
                q2 = q_vals[1] if len(q_vals) > 1 else 0.0
                q3 = q_vals[2] if len(q_vals) > 2 else 0.0
                writer.writerow([s_idx, line, i1, i2, i3, p1, p2, p3, q1, q2, q3])

    # 3. If Fault Dataset, Export Fault Scenario Metadata
    if has_fault_labels:
        print(f"    -> Writing {summary_csv}...")
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario_id", "fault_bus", "fault_type", "fault_resistance"])
            for s_idx, sc in enumerate(scenarios):
                fl = sc.get("fault_label", {})
                writer.writerow([
                    s_idx,
                    fl.get("bus", ""),
                    fl.get("type", ""),
                    fl.get("resistance", 0.0)
                ])

    del scenarios
    gc.collect()
    print(f"[DONE] Completed export for {pkl_path}")


def export_pt_dataset_to_csv(pt_path, output_dir=None, prefix=None):
    """
    Exports PyTorch Geometric tensor dataset (.pt) into node and edge CSV files.
    """
    if not os.path.exists(pt_path):
        print(f"[!] Warning: File {pt_path} does not exist. Skipping.")
        return

    if output_dir is None:
        output_dir = os.path.dirname(pt_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = prefix or os.path.splitext(os.path.basename(pt_path))[0]
    nodes_csv = os.path.join(output_dir, f"{base_name}_nodes.csv")
    edges_csv = os.path.join(output_dir, f"{base_name}_edges.csv")

    print(f"\n[+] Loading PyTorch dataset {pt_path}...")
    dataset = torch.load(pt_path)
    num_samples = len(dataset)
    print(f"    Loaded {num_samples:,} graph samples. Exporting to CSV...")

    # 1. Export Node Features & Target Labels
    print(f"    -> Writing {nodes_csv}...")
    with open(nodes_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "node_idx",
            "v_meas_ph1", "v_meas_ph2", "v_meas_ph3", "v_sensor_mask",
            "v_true_ph1", "v_true_ph2", "v_true_ph3",
            "theta_true_ph1", "theta_true_ph2", "theta_true_ph3"
        ])
        for s_idx, sample in enumerate(tqdm(dataset, desc="    Graph Nodes")):
            node_feats = sample["node_features"].numpy()
            v_true = sample["V_true"].numpy()
            theta_true = sample["theta_true"].numpy()
            num_nodes = node_feats.shape[0]

            for n_idx in range(num_nodes):
                writer.writerow([
                    s_idx, n_idx,
                    node_feats[n_idx, 0], node_feats[n_idx, 1], node_feats[n_idx, 2], node_feats[n_idx, 3],
                    v_true[n_idx, 0], v_true[n_idx, 1], v_true[n_idx, 2],
                    theta_true[n_idx, 0], theta_true[n_idx, 1], theta_true[n_idx, 2]
                ])

    # 2. Export Edge Features, Adjacency, & Target Currents
    print(f"    -> Writing {edges_csv}...")
    with open(edges_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "edge_idx", "source_node", "target_node",
            "resistance_r", "reactance_x",
            "p_meas_ph1", "p_meas_ph2", "p_meas_ph3",
            "q_meas_ph1", "q_meas_ph2", "q_meas_ph3",
            "i_meas_ph1", "i_meas_ph2", "i_meas_ph3",
            "line_sensor_mask",
            "i_true_ph1", "i_true_ph2", "i_true_ph3"
        ])
        for s_idx, sample in enumerate(tqdm(dataset, desc="    Graph Edges")):
            edge_feats = sample["edge_features"].numpy()
            edge_idx = sample["edge_index"].numpy()
            i_true = sample["I_true"].numpy()
            num_edges = edge_feats.shape[0]

            for e_idx in range(num_edges):
                src = edge_idx[0, e_idx]
                dst = edge_idx[1, e_idx]
                writer.writerow([
                    s_idx, e_idx, src, dst,
                    edge_feats[e_idx, 0], edge_feats[e_idx, 1],
                    edge_feats[e_idx, 2], edge_feats[e_idx, 3], edge_feats[e_idx, 4],
                    edge_feats[e_idx, 5], edge_feats[e_idx, 6], edge_feats[e_idx, 7],
                    edge_feats[e_idx, 8], edge_feats[e_idx, 9], edge_feats[e_idx, 10],
                    edge_feats[e_idx, 11],
                    i_true[e_idx, 0], i_true[e_idx, 1], i_true[e_idx, 2]
                ])

    del dataset
    gc.collect()
    print(f"[DONE] Completed export for {pt_path}")


def export_all():
    print("==================================================")
    print("      DATASET TO CSV EXPORT ORCHESTRATOR          ")
    print("==================================================")

    # 1. State Estimation 1k, 10k, 50k, 100k
    state_files = [
        "data/state_estimation/scenarios_1k.pkl",
        "data/state_estimation/scenarios_10k.pkl",
        "data/state_estimation/scenarios_50k.pkl",
        "data/state_estimation/scenarios_100k.pkl"
    ]
    for pkl in state_files:
        export_scenarios_pkl_to_csv(pkl)

    # 2. PyG Tensor Dataset 1k
    export_pt_dataset_to_csv("data/state_estimation/dataset_1k.pt")

    # 3. Fault Scenarios 1k
    export_scenarios_pkl_to_csv("data/fault_localization/fault_scenarios_1k.pkl")

    print("\n==================================================")
    print("      ALL DATASETS SUCCESSFULLY EXPORTED TO CSV    ")
    print("==================================================")


if __name__ == "__main__":
    # Ensure working directory is project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    export_all()
