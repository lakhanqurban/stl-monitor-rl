"""Phase 6: Distribution shift with terrain perturbations (friction/roughness).

Extends existing shift testing to include terrain-based perturbations
and documents a specific failure case.
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import sys, json, time, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.environments.locomotion_wrapper import LocomotionWrapper
from src.training.trajectory_logger import TrajectoryLogger
from src.monitors.stl_monitor import STLSafetyMonitor
import yaml

ENV_ID = "Ant-v5"
OUTPUT_DIR = "data/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("configs/stl_properties.yaml") as f:
    stl_config = yaml.safe_load(f)


class TerrainPerturbedAnt:
    """Apply terrain perturbations to Ant-v5 via MuJoCo geomfriction."""

    def __init__(self, friction_scale=1.0, roughness=0.0):
        self.env = gym.make(ENV_ID, render_mode=None)
        self.wrapped = LocomotionWrapper(self.env)
        self.friction_scale = friction_scale
        self.roughness = roughness
        self._apply_friction()

    def _apply_friction(self):
        model = self.env.unwrapped.model
        for i in range(model.ngeom):
            if model.geom_friction[i, 0] > 0:
                model.geom_friction[i, 0] *= self.friction_scale

    def reset(self, seed=None):
        obs, info = self.wrapped.reset(seed=seed)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.wrapped.step(action)
        if self.roughness > 0:
            obs = obs + np.random.normal(0, self.roughness, obs.shape).astype(obs.dtype)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.wrapped.close()


def run_terrain_condition(model, condition_name, n_episodes=5, **kwargs):
    """Run evaluation under one terrain perturbation."""
    env = TerrainPerturbedAnt(**kwargs)
    monitor = STLSafetyMonitor(stl_config["properties"])

    stl_scores, rewards = [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 100)
        total_reward = 0
        signal_data = {}

        for t in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            mon_obs = info.get("monitor_obs", {})
            readings = {k: float(v[0]) if hasattr(v, '__len__') else float(v)
                        for k, v in mon_obs.items()
                        if k in ["body_roll", "body_pitch", "body_height",
                                 "body_velocity_x", "body_velocity_y"]}
            for k, v in readings.items():
                if k not in signal_data:
                    signal_data[k] = []
                signal_data[k].append(v)

            if terminated or truncated:
                break

        if signal_data and any(len(v) > 0 for v in signal_data.values()):
            stl_results = monitor.evaluate_all_trajectory(signal_data)
            stl_scores.append(monitor.get_safety_score(stl_results))
        rewards.append(total_reward)

    env.close()
    return {
        "condition": condition_name,
        "stl_mean_score": float(np.mean(stl_scores)) if stl_scores else None,
        "stl_std_score": float(np.std(stl_scores)) if stl_scores else None,
        "stl_violation_rate": float(np.mean([s < 0 for s in stl_scores])) if stl_scores else None,
        "mean_reward": float(np.mean(rewards)),
    }


if __name__ == "__main__":
    model_path = None
    for s in [42, 0, 123, 456, 789]:
        p = f"data/checkpoints/seed_{s:03d}/final_model.zip"
        if os.path.exists(p):
            model_path = p
            break
    if not model_path:
        print("No trained model found.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    conditions = [
        ("baseline", {}),
        ("friction_very_low", {"friction_scale": 0.3}),
        ("friction_low", {"friction_scale": 0.6}),
        ("friction_high", {"friction_scale": 1.5}),
        ("roughness_mild", {"roughness": 0.02}),
        ("roughness_moderate", {"roughness": 0.05}),
        ("roughness_extreme", {"roughness": 0.1}),
    ]

    all_results = []
    for name, kwargs in conditions:
        t0 = time.time()
        r = run_terrain_condition(model, name, n_episodes=5, **kwargs)
        all_results.append(r)
        elapsed = time.time() - t0
        stl_str = f"{r['stl_mean_score']:+.3f}" if r['stl_mean_score'] is not None else "N/A"
        print(f"  {name:25s} | STL={stl_str} | Rew={r['mean_reward']:.1f} | {elapsed:.1f}s")

    # Save
    with open(f"{OUTPUT_DIR}/terrain_shift_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Failure case analysis
    print(f"\n{'='*70}")
    print("FAILURE CASE ANALYSIS")
    print(f"{'='*70}")
    baseline_score = all_results[0]["stl_mean_score"]
    for r in all_results[1:]:
        if r["stl_mean_score"] is not None and baseline_score is not None:
            delta = r["stl_mean_score"] - baseline_score
            severity = "MILD" if abs(delta) < 0.2 else "MODERATE" if abs(delta) < 0.5 else "SEVERE"
            print(f"  {r['condition']:25s}: {delta:+.3f} ({severity} degradation)")

    # Table
    print(f"\n{'='*70}")
    print(f"{'Condition':25s} | {'STL Score':>12s} | {'STL Viol':>10s} | {'Reward':>10s}")
    print(f"{'-'*70}")
    for r in all_results:
        stl_s = f"{r['stl_mean_score']:+.3f}±{r['stl_std_score']:.3f}" if r['stl_mean_score'] is not None else "N/A"
        stl_v = f"{r['stl_violation_rate']:.0%}" if r['stl_violation_rate'] is not None else "N/A"
        print(f"{r['condition']:25s} | {stl_s:>12s} | {stl_v:>10s} | {r['mean_reward']:>10.1f}")

    print(f"\nResults saved to {OUTPUT_DIR}/terrain_shift_results.json")
