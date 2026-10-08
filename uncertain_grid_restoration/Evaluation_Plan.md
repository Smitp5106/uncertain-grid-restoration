# Research Evaluation Framework

This document maps out the 6 critical experiments required to validate the proposed Belief-State Reinforcement Learning framework. These experiments translate directly into the "Results" section of a research publication.

## Phase 0: Dataset Generation & Environment Setup
Before conducting the core experiments, a highly stochastic grid environment is simulated to train and evaluate the deep learning models.
- **Simulation Engine:** OpenDSS (serving as the omniscient physical ground truth).
- **Scale:** 10,000 to 100,000 unique simulated scenarios.
- **Stochastic Variables:** 
  - Random load scaling (70% - 130% nominal).
  - Randomized fault locations (SLG, LL, 3P, HIF) and fault resistances.
- **Measurement Corruption:** Ground truth tensors are corrupted via Gaussian noise (Aleatoric) and active sensor dropout masks (e.g., 20% to 80% failure rates) to simulate extreme grid events.

## Phase 1: State Estimation (Supervised Metrics)

### Experiment 1: Physics-Informed Regularization
**Objective:** Prove that baking Kirchhoff's Laws into the loss function improves boundary accuracy.
- **Models:** Pure Data-Driven GNN vs. Physics-Informed GNN
- **Metrics:** Voltage RMSE, Angle RMSE, Current RMSE
- **Hypothesis:** Physics-informed regularization will reduce state-estimation errors, particularly under sparse or unobservable measurement conditions.

### Experiment 2: Uncertainty Decomposition & Calibration
**Objective:** Prove the network reliably bounds its own ignorance and distinguishes data noise from Out-of-Distribution (OOD) model limits.
- **Models:** Point-Estimate GNN vs. Gaussian NLL Uncertainty GNN (with MC Dropout)
- **Metrics:** Calibration, 95% CI Coverage, Negative Log-Likelihood (NLL), $\sigma^2_{epistemic}$, $\sigma^2_{aleatoric}$
- **Hypothesis:** $\sigma^2_{aleatoric}$ will map to base sensor noise limits, while $\sigma^2_{epistemic}$ will reliably increase when predicting deep unobservable or out-of-distribution physical states.

## Phase 2: Fault Diagnosis

### Experiment 3: Probabilistic Fault Localization
**Objective:** Prove the superiority of distributional predictions over rigid classification.
- **Models:** Argmax Fault Classifier vs. Edge-Level Fault Probability GNN
- **Metrics:** Top-1 Accuracy, Top-3 Accuracy, Brier Score (measuring probability calibration)
- **Hypothesis:** The Probability model will yield superior Brier scores, maintaining calibrated probability distributions over topological ambiguity rather than forcing premature singular classifications.

## Phase 3: Restoration Control (RL Metrics)

### Experiment 4: Architectures for Restoration
**Objective:** Validate that native graph convolution outperforms flat vector spaces in dynamic topologies.
- **Models:** Baseline PPO vs. GNN-PPO vs. Uncertainty-GNN-PPO
- **Metrics:** % Load Restored, Total Power Loss, Switch Operation Count
- **Hypothesis:** Graph-based and uncertainty-aware policies may improve restoration efficiency and reduce unnecessary mechanical switching relative to corresponding flat baselines.

### Experiment 5: Robustness to Sparsity (The Core Claim)
**Objective:** Prove the Belief-State RL agent survives sensor loss better than deterministic agents.
- **Variable:** Measurement Availability (100%, 80%, 60%, 40%, 20%)
- **Models:** Deterministic RL vs. Belief-State RL
- **Hypothesis:** The performance degradation of deterministic policies will be greater under severe sensor sparsity, while the belief-state policy will maintain significantly safer operational behavior.

### Experiment 6: Controller Safety under Ambiguity
**Objective:** Test the safety layer and agent conservatism.
- **Case A (High Confidence):** $P(F_7)=0.95, P(F_8)=0.03, P(F_9)=0.02$
- **Case B (Medium Ambiguity):** $P(F_7)=0.60, P(F_8)=0.25, P(F_9)=0.15$
- **Case C (High Ambiguity):** $P(F_7)=0.40, P(F_8)=0.35, P(F_9)=0.25$
- **Metric:** Action Safety Rate, Total Reward
- **Hypothesis:** As fault ambiguity increases (Case C), the agent will increasingly favor conservative or no-op actions to avoid catastrophic safety filter triggers, whereas in high confidence states (Case A), it will act swiftly.

## Phase 4: Final Restoration Algorithm Comparisons

### Experiment 7: Global Baselines
**Objective:** Prove the proposed model outperforms both classical optimization and standard RL baselines.
- **Classical Baselines:** Rule-based heuristics, MILP (Mixed Integer Linear Programming), Meta-heuristics (GA, PSO).
- **ML/RL Baselines:** DQN, Standard PPO, GNN-PPO, GNN-PPO + Safety, GNN-PPO + Uncertainty.
- **Target Model:** **Belief-State Physics-Informed GNN-PPO (YOUR MODEL)**
- **Metrics:** Load restored (%), Energy not served, Power loss, Number of switching operations, Voltage/Thermal violations, Safety-filter rejection rate, Decision time, and Restoration completion rate. (Optimality gap is additionally tracked only against the exact MILP reference solution).

## Phase 5 (Future Work): Topology Scaling (Phase 29)
*Crucially, this progression is only triggered after complete experimental validation on the IEEE 33-bus system.*
- **Level 1:** IEEE 33-Bus (Completed proof of concept)
- **Level 2:** IEEE 69-Bus (Tests scalability of message passing)
- **Level 3:** IEEE 123-Bus (Tests robustness against massive phase imbalances and complex meshing)
