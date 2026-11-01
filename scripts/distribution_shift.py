"""Distribution shift testing: evaluate monitors under perturbations."""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import sys, json, time, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from src.environments.locomotion_wrapper import LocomotionWrapper
from src.training.trajectory_logger import TrajectoryLogger, collect_trajectory
from src.monitors.stl_monitor import STLSafetyMonitor
from src.monitors.baselines import ThresholdMonitor, CBFMonitor
import yaml

ENV_ID = "Ant-v5"
OUTPUT_DIR = "data/results"
TRAJ_DIR = "data/trajectories"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("configs/stl_properties.yaml") as f:
    stl_config = yaml.safe_load(f)


class PerturbedAnt:
    """Apply perturbations to Ant-v5."""

    def __init__(self, force_mag=0.0, noise_std=0.0, mass_scale=1.0):
        self.env = gym.make(ENV_ID, render_mode=None)
        self.wrapped = LocomotionWrapper(self.env)
        self.force_mag = force_mag
        self.noise_std = noise_std
        self.mass_scale = mass_scale
        self._apply_mass()

    def _apply_mass(self):
        if self.mass_scale != 1.0:
            data = self.env.unwrapped.data
            model = self.env.unwrapped.model
            # Scale all body masses
            for i in range(model.nbody):
                model.body_mass[i] *= self.mass_scale

    def reset(self, seed=None):
        obs, info = self.wrapped.reset(seed=seed)
        if self.force_mag > 0:
            self._apply_force()
        if self.noise_std > 0:
            obs = obs + np.random.normal(0, self.noise_std, obs.shape).astype(obs.dtype)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.wrapped.step(action)
        if self.force_mag > 0 and np.random.random() < 0.02:
            self._apply_force()
        if self.noise_std > 0:
            obs = obs + np.random.normal(0, self.noise_std, obs.shape).astype(obs.dtype)
        return obs, reward, terminated, truncated, info

    def _apply_force(self):
        data = self.env.unwrapped.data
        body_id = 1  # torso
        force = np.zeros(6)
        force[:3] = self.force_mag * np.random.randn(3)
        data.xfrc_applied[body_id] = force

    def close(self):
        self.wrapped.close()


def run_condition(model, condition_name, n_episodes=5, **perturb_kwargs):
    """Run evaluation under one perturbation condition."""
    env = PerturbedAnt(**perturb_kwargs)
    monitor = STLSafetyMonitor(stl_config["properties"])
    tm = ThresholdMonitor(
        {"body_roll": 1.0, "body_pitch": 1.0, "body_height": 0.4},
        {"body_roll": "both", "body_pitch": "both", "body_height": "lower"},
    )
    cbf = CBFMonitor(6, {"body_roll": (-1.0, 1.0), "body_pitch": (-1.0, 1.0), "body_height": (0.4, 5.0)})

    stl_scores, t_vr, c_vr, rewards = [], [], [], []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 100)
        total_reward = 0
        logger = TrajectoryLogger()

        for t in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            mon_obs = info.get("monitor_obs", {})
            logger.log_step(t, mon_obs, reward, terminated or truncated)
            if terminated or truncated:
                break

        # Evaluate monitors on this episode
        if logger.data["timestep"]:
            import pandas as pd
            df = pd.DataFrame(logger.data)
            sd = logger.to_signal_data(df)
            stl_results = monitor.evaluate_all_trajectory(sd)
            stl_scores.append(monitor.get_safety_score(stl_results))
            t_vr.append(tm.check_trajectory(sd)["violation_rate"])
            c_vr.append(cbf.check_trajectory(sd)["violation_rate"])
        rewards.append(total_reward)

    env.close()
    return {
        "condition": condition_name,
        "stl_mean_score": float(np.mean(stl_scores)) if stl_scores else None,
        "stl_std_score": float(np.std(stl_scores)) if stl_scores else None,
        "stl_violation_rate": float(np.mean([s < 0 for s in stl_scores])) if stl_scores else None,
        "threshold_violation_rate": float(np.mean(t_vr)) if t_vr else None,
        "cbf_violation_rate": float(np.mean(c_vr)) if c_vr else None,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


if __name__ == "__main__":
    # Load one trained model (seed 42 — mid-range performer)
    model_path = "data/checkpoints/seed_042/final_model"
    if not os.path.exists(model_path):
        # Try any available model
        for s in [0, 42, 123, 456, 789]:
            p = f"data/checkpoints/seed_{s:03d}/final_model"
            if os.path.exists(p):
                model_path = p
                break
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    conditions = [
        ("baseline", {}),
        ("force_small", {"force_mag": 5.0}),
        ("force_medium", {"force_mag": 15.0}),
        ("force_large", {"force_mag": 30.0}),
        ("noise_mild", {"noise_std": 0.01}),
        ("noise_moderate", {"noise_std": 0.05}),
        ("noise_severe", {"noise_std": 0.15}),
        ("mass_light", {"mass_scale": 0.8}),
        ("mass_heavy", {"mass_scale": 1.3}),
        ("mass_extreme", {"mass_scale": 1.6}),
    ]

    all_results = []
    for name, kwargs in conditions:
        t0 = time.time()
        r = run_condition(model, name, n_episodes=5, **kwargs)
        all_results.append(r)
        elapsed = time.time() - t0
        print(f"  {name:20s} | STL={r['stl_mean_score']:+.3f} | Thr={r['threshold_violation_rate']:.0%} | CBF={r['cbf_violation_rate']:.0%} | Rew={r['mean_reward']:.1f} | {elapsed:.1f}s")

        with open(f"{OUTPUT_DIR}/shift_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Condition':20s} | {'STL Score':>12s} | {'STL Viol':>10s} | {'Thr Viol':>10s} | {'CBF Viol':>10s} | {'Reward':>10s}")
    print(f"{'-'*80}")
    for r in all_results:
        stl_s = f"{r['stl_mean_score']:+.3f}±{r['stl_std_score']:.3f}" if r['stl_mean_score'] is not None else "N/A"
        stl_v = f"{r['stl_violation_rate']:.0%}" if r['stl_violation_rate'] is not None else "N/A"
        thr_v = f"{r['threshold_violation_rate']:.0%}" if r['threshold_violation_rate'] is not None else "N/A"
        cbf_v = f"{r['cbf_violation_rate']:.0%}" if r['cbf_violation_rate'] is not None else "N/A"
        print(f"{r['condition']:20s} | {stl_s:>12s} | {stl_v:>10s} | {thr_v:>10s} | {cbf_v:>10s} | {r['mean_reward']:>10.1f}")

    print(f"\nResults saved to {OUTPUT_DIR}/shift_results.json")
