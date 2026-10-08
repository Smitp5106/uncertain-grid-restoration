# System Architecture & Implementation Details

This document provides a comprehensive, deep-dive breakdown of the modules implemented in the **Uncertain Grid Restoration** project. It outlines the mathematical theory and the exact Python/PyTorch logic used to build the framework.

---

## 1. Physics Simulator & Data Pipeline (`simulator/`)
To train robust Machine Learning models, we bypass static datasets and actively simulate the power grid using the `OpenDSS` engine.

### Data Generation Steps:
1. **Scenario Generation (`scenario_generator.py`)**: Modifies the base IEEE-33 bus topology by randomizing load multipliers (70% - 130%) and injecting Distributed Energy Resources (DERs).
2. **Fault Simulation (`fault_generator.py`)**: Iterates through the grid and injects diverse faults (Single-Line-to-Ground, Line-to-Line, 3-Phase, High Impedance) with randomized fault resistances.
3. **Omniscient State Extraction (`extract_state.py`)**: Solves the power flow and extracts the absolute "Ground Truth" tensors ($V_{true}, \theta_{true}, I_{true}$).
4. **Measurement Masking (`measurement_generator.py`)**: To simulate the real world, the ground truth is corrupted. Gaussian noise is added to the floats, and a uniform dropout mask (e.g., 20% missing sensors) sets unobservable values to $0.0$, accompanied by an explicit `mask = 0` integer.

### Dataset Compilation (`build_dataset.py`)
The raw scenarios are compiled into PyTorch Geometric (`.pt`) datasets. The dataset strictly separates the corrupted input (`X`, `E`) from the target labels (`V_true`, `theta_true`).

---

## 2. Graph Neural Networks for State Estimation (`models/`)
The core of the framework is estimating the grid state despite missing sensors.

### Physics-Informed GNN (`physics_gnn.py`)
A standard GNN using Mean Squared Error (MSE) flatlines when 40% of sensors go dark. To fix this, we implemented a custom loss function:
$$ L = L_{data} + \lambda_1 L_{KCL} + \lambda_2 L_{Power} + \lambda_3 L_{Voltage} $$
- **KCL Loss**: Penalizes the model if the predicted currents entering a bus do not equal the currents leaving it.
- **Voltage Loss**: Heavily penalizes predicted voltages outside the safety bounds ($V < 0.95$ or $V > 1.05$).

### Epistemic & Aleatoric Uncertainty (`uncertainty_gnn.py`)
The network is upgraded to output both a mean ($\mu$) and a variance ($\sigma$). 
- **Negative Log-Likelihood (NLL)**: The network is trained using Gaussian NLL. High uncertainty bounds ($\sigma$) are output for missing sensors.
- **Monte Carlo Dropout**: Implemented via the `mc_dropout_predict()` function. By leaving dropout active during inference and sampling the network $N$ times, the framework decomposes total uncertainty into:
  - **Aleatoric Uncertainty**: Uncertainty associated with measurement/data noise ($\sigma^2_{aleatoric} \approx \frac{1}{N}\sum_i\sigma_i^2$).
  - **Epistemic Uncertainty**: Uncertainty associated with model knowledge, which can increase in poorly observed or out-of-distribution conditions ($\sigma^2_{epistemic} \approx Var(\mu_1,\mu_2,\ldots,\mu_N)$).

---

## 3. Probabilistic Fault Localization (`fault_gnn.py`)
Rather than outputting a single classification (e.g., "Fault = Line 17"), the Fault GNN executes an **Edge-Level Readout**.
- For every line connecting node $u$ and $v$, it concatenates their embeddings and passes them through an MLP.
- **Output**: A tensor of shape `[num_edges, 1]`, representing the exact probability $P(\text{fault})$ for every line. 

---

## 4. The Belief Graph (`graph/graph_features.py`)
The RL agent requires a Markov state. We merge the outputs of the Uncertainty GNN and Fault Localization GNN into a single **Belief Graph**.
- **Node Features (`[N, 8]`)**: `[V_hat, theta_hat, sigma_V, sigma_theta, P_load, Q_load, P_generation, sensor_mask]`
- **Edge Features (`[E, 6]`)**: `[R, X, I_hat, I_max, switch_status, fault_probability]`

---

## 5. Safe Reinforcement Learning Control (`environment/`)
The final module is an autonomous restoration agent trained via Proximal Policy Optimization (PPO).

### The Environment (`restoration_env.py`)
Built on the `Gymnasium` API. The environment controls OpenDSS.
- **Action Space**: Defined dynamically as $|\mathcal A| = N_{switch} + 1$, where $0$ is a "No-Op", and $\{1, \dots, N\}$ toggles the corresponding switch. This inherently scales from IEEE-33 to massive IEEE-123 systems without hardcoding `Open/Close` pairs.
- **Observation Space**: `spaces.Dict` representing the Belief Graph.

### Physics Safety Filter (`safety/safety_filter.py`)
An interceptor layer preventing destructive actions by enforcing strict grid logic:
1. **Radiality & Islanding (`networkx`)**: Before running power flow, the candidate graph is checked for cycles (`nx.find_cycle`) and islanding (`nx.is_connected`). If it is not a mathematically pure, energized spanning tree, it is rejected.
2. **Convergence**: OpenDSS trial power flow is solved. If it fails to converge, the state is rolled back.
3. **Voltage Limits**: Verifies all energized bus voltages fall within $[0.95, 1.05]$ PU.
4. **Thermal Limits**: Verifies no line current exceeds its physical $I_{max}$ ampacity rating.
If any of these bounds are breached, the action is rejected and the simulator reverts.

### Reward Function
The agent is trained strictly on physics-based rewards. The step reward is formulated as:
$$ R = 10 \cdot R_{load} - 1.0 \cdot R_{loss} - 0.1 \cdot R_{switch} - 20 \cdot R_{violation} $$

Where the individual components are explicitly defined as:
- **$R_{load}$**: $\frac{P_{served}}{P_{total}}$, the percentage of total active power load currently energized and served by the grid.
- **$R_{loss}$**: Total $I^2R$ active power loss in the grid (normalized to a Per-Unit base to match gradient scaling).
- **$R_{switch}$**: $1$ if a switching operation occurred in the current step (penalizing excessive mechanical wear), else $0$.
- **$R_{violation}$**: $1$ if the action was rejected by the Physics Safety Filter (preventing reckless agent exploration), else $0$.

### Training Architectures (`training/train_rl.py`)
- **Baseline PPO**: Flattens the Belief Graph nodes and edges into a massive 1D vector and processes it via standard MLPs.
- **GNN-PPO**: A custom `GraphGNNExtractor` dynamically handles batching inside SB3, utilizing native PyTorch Geometric `GATv2Conv` layers to process the Belief Graph via Message Passing before feeding it to the PPO Actor-Critic.
