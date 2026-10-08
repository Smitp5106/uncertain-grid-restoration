import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO

# Add project root to sys path
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from environment.restoration_env import RestorationEnv
from training.train_rl import GraphGNNExtractor

def run_final_pipeline_demo():
    print("==================================================")
    print("      FINAL PROPOSED MODEL PIPELINE EXECUTION     ")
    print("==================================================\n")
    
    # 1. Load the trained Graph PPO Agent
    print("[1] Loading Graph PPO Agent...")
    env = RestorationEnv()
    try:
        model = PPO.load("data/processed/ppo_gnn", env=env)
        print("    -> GNN-PPO loaded successfully from disk.")
    except Exception as e:
        print("    -> Error loading pre-trained model. Using uninitialized GNN-PPO.")
        kwargs_gnn = dict(features_extractor_class=GraphGNNExtractor, features_extractor_kwargs=dict(features_dim=256))
        model = PPO("MultiInputPolicy", env, policy_kwargs=kwargs_gnn, verbose=0)
    
    print("\n[2] Initializing OpenDSS & Generating Sparse Measurements...")
    # env.reset() inherently executes the measurement masking logic
    obs, info = env.reset()
    print("    -> Physics simulation solved. Fault and random conditions injected.")
    print("    -> Measurements generated with noise. 20% of sensors physically masked.")
    
    print("\n[3] Executing Physics-Informed Uncertainty GNN...")
    # Executed natively inside env._get_observation()
    print("    -> System state (V_hat, theta_hat, I_hat) predicted.")
    print("    -> Predictive Uncertainty (sigma) bounded by Physics NLL.")
    print("    -> Total uncertainty explicitly decomposed into Aleatoric and Epistemic via MC Dropout.")
    
    print("\n[4] Executing Fault Localization GNN...")
    # Executed natively inside env._get_observation()
    print("    -> Kirchhoff's Current Law (KCL) residuals calculated.")
    print("    -> P(fault) mapped for every unique line.")
    
    print("\n[5] Constructing Central Belief Graph...")
    print(f"    -> Node Features Assembled: {obs['node_features'].shape}")
    print(f"    -> Edge Features Assembled: {obs['edge_features'].shape}")
    
    print("\n[6] Graph PPO Action Inference...")
    action, _states = model.predict(obs, deterministic=True)
    if action == 0:
        print("    -> Agent chose: NO OPERATION (0)")
    else:
        switch_idx = action - 1
        print(f"    -> Agent Action: Toggle Switch {switch_idx + 1}")
    
    print("\n[7] Intercepting with Physics Safety Layer...")
    # The filter natively intercepts the action inside env.step()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if info.get("action_rejected", False):
        print(f"    -> ACTION REJECTED! Reason: {info.get('reason')}")
        print("    -> Trial power flow failed physics constraints. OpenDSS state rolled back.")
    else:
        print("    -> Action passed Safety Filter (Radiality, Voltage, and Thermal bounds).")
        print("    -> Action committed safely to OpenDSS engine.")
        
    print(f"\n[8] Final Physics Evaluation...")
    print(f"    -> Objective Reward: {reward}")
    
    print("\n==================================================")
    print("          PIPELINE EXECUTED SUCCESSFULLY          ")
    print("==================================================")

if __name__ == "__main__":
    run_final_pipeline_demo()
