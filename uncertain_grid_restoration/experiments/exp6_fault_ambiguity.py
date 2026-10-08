import os
import sys
import torch
import numpy as np
from stable_baselines3 import PPO

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

from environment.restoration_env import RestorationEnv

def test_ambiguity_case(case_name, fault_probs, model, env):
    print(f"\n--- Running {case_name} ---")
    print(f"Injected Fault Probabilities (Lines 7, 8, 9): {fault_probs}")
    
    obs, info = env.reset()
    
    # Artificially override the fault probabilities in the edge features 
    # (Index 5 in edge features is fault_probability)
    # The SB3 observation is flat, so we mock it for the agent's observation:
    
    # Edge features shape is (64, 6) in the raw dict
    # Lines 7, 8, and 9 correspond roughly to physical switches. 
    # We inject the probability array across those targeted edges.
    edge_feats = obs["edge_features"]
    
    # Zero out all faults
    edge_feats[:, 5] = 0.01 
    
    # Inject ambiguity
    edge_feats[6, 5] = fault_probs[0] # Line 7
    edge_feats[7, 5] = fault_probs[1] # Line 8
    edge_feats[8, 5] = fault_probs[2] # Line 9
    
    # Feed to agent
    action, _states = model.predict(obs, deterministic=True)
    
    if action == 0:
        print("Agent Action: NO OPERATION (Acted Conservatively)")
    else:
        switch_idx = action - 1
        print(f"Agent Action: Toggle Switch {switch_idx + 1}")
        
    # Evaluate safety
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    if info.get("action_rejected", False):
        print("Safety Evaluation: ACTION REJECTED BY SAFETY FILTER (UNSAFE)")
    else:
        print("Safety Evaluation: ACTION PASSED PHYSICS FILTER (SAFE)")
    print(f"Reward Received: {reward:.2f}")

def run_experiment_6():
    print("==================================================")
    print(" EXPERIMENT 6: CONTROLLER SAFETY UNDER AMBIGUITY  ")
    print("==================================================")
    
    env = RestorationEnv()
    
    # Load trained model (fallback to initialized model if not found)
    try:
        model = PPO.load("data/processed/ppo_gnn", env=env)
    except:
        from training.train_rl import GraphGNNExtractor
        kwargs_gnn = dict(features_extractor_class=GraphGNNExtractor, features_extractor_kwargs=dict(features_dim=256))
        model = PPO("MultiInputPolicy", env, policy_kwargs=kwargs_gnn, verbose=0)
        
    # Case A: High Confidence
    test_ambiguity_case("Case A (High Confidence)", [0.95, 0.03, 0.02], model, env)
    
    # Case B: Medium Ambiguity
    test_ambiguity_case("Case B (Medium Ambiguity)", [0.60, 0.25, 0.15], model, env)
    
    # Case C: High Ambiguity
    test_ambiguity_case("Case C (High Ambiguity)", [0.40, 0.35, 0.25], model, env)
    
    print("\n==================================================")

if __name__ == "__main__":
    run_experiment_6()
