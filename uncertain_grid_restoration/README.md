# Uncertain Grid Restoration

A physics-informed, belief-state Reinforcement Learning (RL) framework for active distribution network restoration under severe partial observability and sensor failure.

## Overview
This repository implements a complete pipeline to restore power distribution grids (starting with IEEE-33) following faults. Because real-world grid sensors can be noisy or unavailable during extreme events, restoration agents operating on deterministic state estimates may become less reliable under partial observability.

This framework solves this by:
1. **State Estimation**: A Physics-Informed GNN estimates the hidden grid state ($V, \theta, I$) from sparse measurements, constrained by Kirchhoff's Laws.
2. **Uncertainty-Aware State Estimation**: The GNN predicts the state mean ($\mu$) and predictive uncertainty ($\sigma$) using Gaussian Negative Log-Likelihood. Sensor masking is used to evaluate whether uncertainty increases as observability decreases.
3. **Uncertainty Decomposition**: Using Monte Carlo (MC) Dropout at inference time, the total variance is decomposed into $\sigma^2_{total} = \sigma^2_{aleatoric} + \sigma^2_{epistemic}$, allowing the agent to explicitly distinguish between inherent measurement/data noise (aleatoric) and model knowledge limits in poorly observed or out-of-distribution conditions (epistemic).
4. **Probabilistic Fault Localization**: A secondary GNN uses physics residuals (KCL mismatch) to assign fault probabilities to every line in the grid.
5. **Belief Graph**: These predictions merge into a unified "Belief Graph".
6. **Safe Reinforcement Learning**: A Graph-PPO agent navigates this Belief Graph to flip switches. A strict topological and physical **Safety Filter** guarantees the agent cannot physically damage the grid during exploration.

---

## Project Architecture

```text
uncertain_grid_restoration/
│
├── data/                       # Datasets and processed ML tensors
│   ├── raw/
│   ├── state_estimation/
│   ├── fault_localization/
│   └── processed/
│
├── dss/                        # OpenDSS physics engine files
│   └── ieee33/
│       ├── Master.dss
│       ├── Lines.dss
│       └── ...
│
├── simulator/                  # Grid Physics & Data Generation
│   ├── run_dss.py              # OpenDSS solver validation
│   ├── extract_state.py        # Extracts raw V, theta, I from OpenDSS
│   ├── scenario_generator.py   # Generates nominal scenarios (Load/DER variations)
│   ├── fault_generator.py      # Injects SLG, LL, 3P, HIF faults
│   ├── measurement_generator.py# Adds Gaussian noise and random sensor masking
│   └── build_dataset.py        # Compiles raw scenarios into PyTorch Geometric tensors
│
├── graph/                      # PyTorch Geometric Transformations
│   ├── build_graph.py          # Extracts directed/undirected topology from OpenDSS
│   └── graph_features.py       # Assembles the Belief Graph for the RL agent
│
├── models/                     # Deep Learning Architectures
│   ├── gnn_baseline.py         # Pure data-driven GNN
│   ├── physics_gnn.py          # GNN with KCL, Voltage, and Power loss penalties
│   ├── uncertainty_gnn.py      # Predicts mu and sigma via NLL
│   └── fault_gnn.py            # Predicts P(fault) per-line using physics residuals
│
├── safety/                     # Action Interceptors
│   └── safety_filter.py        # Rejects RL actions that cause loops or thermal/voltage violations
│
├── environment/                # Reinforcement Learning Environment
│   └── restoration_env.py      # Gymnasium environment executing the full Belief State pipeline
│
├── training/                   # Model Optimization
│   ├── train_state.py          # Trains the State Estimation GNNs
│   ├── train_fault.py          # Trains the Fault Localization GNN
│   └── train_rl.py             # Trains the Graph-PPO RL Agent
│
├── experiments/                # Research Evaluation Scripts
│   ├── exp6_fault_ambiguity.py # Evaluates RL safety under varying fault probabilities
│   └── ...
│
├── demo_pipeline.py            # Master execution script tracing the entire framework
└── Evaluation_Plan.md          # Formal mapping of the 6 core research experiments
```

---

## Theoretical Foundation

### 1. Physics-Informed Loss ($L_{Physics}$)
Instead of pure MSE, the models are trained using a hybrid objective:
$$ L = L_{data} + \lambda_1 L_{KCL} + \lambda_2 L_{Power} + \lambda_3 L_{Voltage} $$
- **$L_{KCL}$**: Penalizes non-zero divergence of predicted line currents at internal nodes ($\sum \hat{I}_{in} - \sum \hat{I}_{out} = 0$).
- **$L_{Voltage}$**: Heavily penalizes predicted voltages outside safety bounds ($< 0.95$ pu or $> 1.05$ pu).

### 2. Belief State Representation
The RL agent does not receive deterministic ground truths. It receives a `Belief Graph` containing:
- **Node Features**: `[V_mean, theta_mean, V_sigma, theta_sigma, P_load, Q_load, P_gen, sensor_mask]`
- **Edge Features**: `[R, X, I_mean, I_max, switch_status, P(fault)]`

### 3. Reward Formulation
The RL agent is taught grid physics via dynamic environment feedback:
$$ R = w_1 R_{load} - w_2 R_{loss} - w_3 R_{switch} - w_4 R_{violation} $$
(e.g., heavily rewarded for restoring loads, massively penalized for forming non-radial loops).

---

## Getting Started

### 1. Verify Grid Physics
Ensure OpenDSS solves the IEEE 33-bus system natively:
```bash
python simulator/run_dss.py
```

### 2. Generate Datasets
Generate massive stochastic variations of the grid (both healthy and faulty):
```bash
python simulator/scenario_generator.py
python simulator/fault_generator.py
python simulator/measurement_generator.py
python simulator/build_dataset.py
```

### 3. Train & Validate GNNs
Train the state estimator and validate that it correctly outputs high uncertainty (large $\sigma$) when sensors are masked:
```bash
python training/train_state.py
python evaluation/state_metrics.py
```

### 4. Run the Full Demo Pipeline
Execute a single simulated step tracing data from the noisy sensors through the physics GNNs, into the Belief Graph, out to the RL agent, and finally through the Safety Filter:
```bash
python demo_pipeline.py
```

### 5. Train the Restoration RL Agent
Kick off the PPO training loop:
```bash
python training/train_rl.py
```

## Future Scaling
Once the models are rigorously validated against the baselines defined in `Evaluation_Plan.md` on IEEE-33, the architecture is designed to scale natively to **IEEE-69** and **IEEE-123** bus systems by simply swapping the OpenDSS Master file.
