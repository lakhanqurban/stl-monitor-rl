"""Predictive vs Reactive STL Monitoring Comparison.

Compares reactive (observed-only) vs predictive (world model + observed)
STL evaluation on locomotion trajectories.

Usage:
    python scripts/predictive_stl_monitor.py --env Ant-v5
"""

import os
import sys
import warnings
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from stable_baselines3 import PPO
from src.monitors.stl_monitor import STLSafetyMonitor
from src.environments.locomotion_wrapper import ENV_CONFIGS
from scripts.world_model_train import WorldModel, PredictiveSTLMonitor


def compare_monitoring(env_name, prediction_horizon=20, max_steps=200):
    """Run reactive vs predictive monitoring and compare."""
    config = ENV_CONFIGS.get(env_name, ENV_CONFIGS['Ant-v5'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load STL properties
    config_file = PROJECT_ROOT / 'configs' / (
        'stl_properties_halfcheetah.yaml' if config.get('is_2d') else 'stl_properties.yaml'
    )
    with open(config_file) as f:
        stl_config = yaml.safe_load(f)

    stl_monitor = STLSafetyMonitor(stl_config['properties'])

    # Load world model
    wm_path = PROJECT_ROOT / 'data' / f'world_model_{env_name}.pt'
    if wm_path.exists():
        checkpoint = torch.load(wm_path, map_location=device, weights_only=False)
        obs_dim = checkpoint['obs_dim']
        action_dim = checkpoint['action_dim']
        world_model = WorldModel(obs_dim, action_dim, latent_dim=128).to(device)
        world_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded world model from {wm_path}")
    else:
        print(f"No world model found at {wm_path}. Run world_model_train.py first.")
        return

    predictive_monitor = PredictiveSTLMonitor(world_model, stl_monitor, device)

    # Load environment and policy
    env = gym.make(env_name, render_mode=None)
    model_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'seed_000' / 'final_model'
    if config.get('is_2d'):
        model_path = PROJECT_ROOT / 'data' / 'checkpoints_halfcheetah' / 'seed_000' / 'final_model'
    model = PPO.load(str(model_path))

    print(f"\n=== Reactive vs Predictive STL Monitoring: {env_name} ===")
    print(f"Prediction horizon: {prediction_horizon} steps")
    print(f"{'Step':>6} {'Reactive':>10} {'Predictive':>10} {'Delta':>8} {'Benefit':>10}")
    print("-" * 50)

    obs, _ = env.reset()
    reactive_scores = []
    predictive_scores = []
    action_buffer = []

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        action_buffer.append(action)

        # Reactive: evaluate STL on current observation
        readings = {
            'body_height': float(env.unwrapped.data.xpos[config['torso_id'], 2]),
            'body_roll': 0.0,
            'body_pitch': 0.0,
            'body_velocity_x': float(env.unwrapped.data.qvel[0]),
            'n_airborne_feet': 2.0,
        }

        reactive_results = {}
        for prop_name in stl_monitor.monitors:
            reactive_results[prop_name] = stl_monitor.evaluate_online_step(
                prop_name, step, readings
            )
        reactive_score = stl_monitor.get_safety_score(reactive_results)
        reactive_scores.append(reactive_score)

        # Predictive: use world model to predict future and evaluate STL
        if len(action_buffer) >= prediction_horizon:
            recent_actions = np.array(action_buffer[-prediction_horizon:])
            action_tensor = torch.FloatTensor(recent_actions).unsqueeze(0).to(device)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            pred_result = predictive_monitor.predict_and_evaluate(
                obs_tensor, action_tensor, horizon=prediction_horizon
            )
            pred_score = pred_result['predictive_score']
        else:
            pred_score = reactive_score

        predictive_scores.append(pred_score)

        delta = pred_score - reactive_score
        benefit = "EARLY" if delta < -0.1 else "SAME"

        if step % 20 == 0:
            print(f"{step:>6} {reactive_score:>10.4f} {pred_score:>10.4f} {delta:>+8.4f} {benefit:>10}")

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            action_buffer = []

    env.close()

    # Summary
    print("-" * 50)
    avg_reactive = np.mean(reactive_scores)
    avg_predictive = np.mean(predictive_scores)
    n_early = sum(1 for r, p in zip(reactive_scores, predictive_scores) if p < r - 0.05)
    print(f"Average reactive score:   {avg_reactive:.4f}")
    print(f"Average predictive score:  {avg_predictive:.4f}")
    print(f"Early warnings (pred < reactive - 0.05): {n_early}/{len(reactive_scores)} ({100*n_early/max(1,len(reactive_scores)):.1f}%)")
    print(f"\nPredictive monitoring provides {n_early/max(1,len(reactive_scores))*100:.1f}% early detection opportunities")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Ant-v5')
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    compare_monitoring(args.env, args.horizon, args.steps)
