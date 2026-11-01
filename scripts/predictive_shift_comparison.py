"""Distribution Shift Predictive vs Reactive STL Comparison.

Usage:
    python scripts/predictive_shift_comparison.py
"""

import os, sys, warnings, yaml, json
import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gymnasium as gym
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitors.stl_monitor import STLSafetyMonitor
from src.environments.locomotion_wrapper import ENV_CONFIGS
from scripts.world_model_train import WorldModel, PredictiveSTLMonitor


def run_comparison(env_name, perturbation_type='mass', severities=None,
                   prediction_horizon=20, n_episodes=3, steps_per_episode=200):
    if severities is None:
        severities = [1.0, 1.3, 1.6]

    config = ENV_CONFIGS.get(env_name, ENV_CONFIGS['Ant-v5'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_file = PROJECT_ROOT / 'configs' / (
        'stl_properties_halfcheetah.yaml' if config.get('is_2d') else 'stl_properties.yaml'
    )
    with open(config_file) as f:
        stl_config = yaml.safe_load(f)

    stl_monitor = STLSafetyMonitor(stl_config['properties'])

    wm_path = PROJECT_ROOT / 'data' / f'world_model_{env_name}.pt'
    checkpoint = torch.load(wm_path, map_location=device, weights_only=False)
    world_model = WorldModel(checkpoint['obs_dim'], checkpoint['action_dim'], latent_dim=128).to(device)
    world_model.load_state_dict(checkpoint['model_state_dict'])
    world_model.eval()
    predictive_monitor = PredictiveSTLMonitor(world_model, stl_monitor, device)

    env = gym.make(env_name, render_mode=None)
    base_mass = env.unwrapped.model.body_mass.copy()

    model_path = PROJECT_ROOT / 'data' / 'checkpoints' / 'seed_000' / 'final_model'
    if config.get('is_2d'):
        model_path = PROJECT_ROOT / 'data' / 'checkpoints_halfcheetah' / 'seed_000' / 'final_model'
    model = PPO.load(str(model_path))

    results = {}

    for severity in severities:
        print(f"\n--- {perturbation_type} severity={severity} ---")
        reactive_scores = []
        predictive_scores = []
        action_buffer = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            env.unwrapped.model.body_mass[:] = base_mass.copy()

            if perturbation_type == 'mass':
                env.unwrapped.model.body_mass[:] = base_mass * severity

            action_buffer = []
            data = env.unwrapped.data

            for step in range(steps_per_episode):
                action, _ = model.predict(obs, deterministic=True)
                action_buffer.append(action)

                readings = {
                    'body_height': float(data.xpos[config['torso_id'], 2]),
                    'body_roll': 0.0,
                    'body_pitch': 0.0,
                    'body_velocity_x': float(data.qvel[0]),
                    'n_airborne_feet': 2.0,
                }
                reactive_results = {}
                for prop_name in stl_monitor.monitors:
                    reactive_results[prop_name] = stl_monitor.evaluate_online_step(prop_name, step, readings)
                reactive_score = stl_monitor.get_safety_score(reactive_results)

                if np.isnan(reactive_score) or np.isinf(reactive_score):
                    reactive_score = -10.0
                reactive_scores.append(reactive_score)

                if len(action_buffer) >= prediction_horizon:
                    recent = np.array(action_buffer[-prediction_horizon:])
                    at = torch.FloatTensor(recent).unsqueeze(0).to(device)
                    ot = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    pred_result = predictive_monitor.predict_and_evaluate(ot, at, horizon=prediction_horizon)
                    pred_score = pred_result['predictive_score']
                    if np.isnan(pred_score) or np.isinf(pred_score):
                        pred_score = -10.0
                else:
                    pred_score = reactive_score
                predictive_scores.append(pred_score)

                if perturbation_type == 'noise':
                    obs = obs + np.random.normal(0, severity, obs.shape)

                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
                    if perturbation_type == 'mass':
                        env.unwrapped.model.body_mass[:] = base_mass * severity
                    action_buffer = []

        avg_r = np.mean([s for s in reactive_scores if np.isfinite(s)])
        avg_p = np.mean([s for s in predictive_scores if np.isfinite(s)])
        early = sum(1 for r, p in zip(reactive_scores, predictive_scores)
                    if np.isfinite(r) and np.isfinite(p) and p < r - 0.05)
        total = len([s for s in reactive_scores if np.isfinite(s)])

        results[str(severity)] = {
            'reactive': avg_r, 'predictive': avg_p,
            'delta': avg_p - avg_r, 'early_pct': 100 * early / max(1, total)
        }
        print(f"  Reactive:   {avg_r:.4f}")
        print(f"  Predictive: {avg_p:.4f}")
        print(f"  Early warnings: {100*early/max(1,total):.1f}%")

    env.close()
    return results


if __name__ == '__main__':
    all_results = {}

    print("=" * 60)
    print("Ant-v5: Mass Variation")
    print("=" * 60)
    all_results['ant_mass'] = run_comparison('Ant-v5', 'mass', [0.8, 1.0, 1.3])

    print("\n" + "=" * 60)
    print("HalfCheetah-v5: Mass Variation")
    print("=" * 60)
    all_results['hc_mass'] = run_comparison('HalfCheetah-v5', 'mass', [0.8, 1.0, 1.3])

    save_path = PROJECT_ROOT / 'data' / 'predictive_shift_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")
