"""Distribution shift experiments with per-property and temporal tracking.

Collects:
1. Per-property violation rates (for the property-level table)
2. Per-timestep robustness traces (for temporal robustness plots)
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import sys, json, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from src.environments.locomotion_wrapper import LocomotionWrapper
from src.training.trajectory_logger import TrajectoryLogger
from src.monitors.stl_monitor import STLSafetyMonitor
from src.monitors.baselines import ThresholdMonitor, CBFMonitor
import yaml

ENV_ID = "Ant-v5"
OUTPUT_DIR = "data/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("configs/stl_properties.yaml") as f:
    stl_config = yaml.safe_load(f)

PROPERTY_NAMES = list(stl_config["properties"].keys())


class PerturbedAnt:
    def __init__(self, force_mag=0.0, noise_std=0.0, mass_scale=1.0):
        self.env = gym.make(ENV_ID, render_mode=None)
        self.wrapped = LocomotionWrapper(self.env)
        self.force_mag = force_mag
        self.noise_std = noise_std
        self.mass_scale = mass_scale
        self._apply_mass()

    def _apply_mass(self):
        if self.mass_scale != 1.0:
            model = self.env.unwrapped.model
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
        body_id = 1
        force = np.zeros(6)
        force[:3] = self.force_mag * np.random.randn(3)
        data.xfrc_applied[body_id] = force

    def close(self):
        self.wrapped.close()


def run_condition(model, condition_name, n_episodes=5, **perturb_kwargs):
    """Run evaluation and collect per-property + per-timestep data."""
    env = PerturbedAnt(**perturb_kwargs)
    monitor = STLSafetyMonitor(stl_config["properties"])
    tm = ThresholdMonitor(
        {"body_roll": 1.0, "body_pitch": 1.0, "body_height": 0.4},
        {"body_roll": "both", "body_pitch": "both", "body_height": "lower"},
    )
    cbf = CBFMonitor(6, {"body_roll": (-1.0, 1.0), "body_pitch": (-1.0, 1.0), "body_height": (0.4, 5.0)})

    stl_scores, t_vr, c_vr, rewards = [], [], [], []
    # Per-property tracking: list of dicts, one per episode
    per_property_episodes = []
    # Temporal traces: list of arrays, one per episode
    temporal_traces_episodes = []

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

        if logger.data["timestep"]:
            df = pd.DataFrame(logger.data)
            sd = logger.to_signal_data(df)
            stl_results = monitor.evaluate_all_trajectory(sd)
            stl_scores.append(monitor.get_safety_score(stl_results))
            t_vr.append(tm.check_trajectory(sd)["violation_rate"])
            c_vr.append(cbf.check_trajectory(sd)["violation_rate"])

            # Per-property: final robustness and whether satisfied
            prop_violations = {}
            for pname, pres in stl_results.items():
                if pres.get("robustness") is not None:
                    prop_violations[pname] = {
                        "robustness": pres["robustness"],
                        "satisfied": pres["satisfied"],
                    }
                else:
                    prop_violations[pname] = {"robustness": None, "satisfied": False}
            per_property_episodes.append(prop_violations)

            # Temporal traces: per-timestep robustness for each property
            prop_traces = {}
            for pname, pres in stl_results.items():
                if "trace" in pres and pres["trace"] is not None:
                    prop_traces[pname] = pres["trace"]
                else:
                    prop_traces[pname] = []
            temporal_traces_episodes.append(prop_traces)
        else:
            per_property_episodes.append({})
            temporal_traces_episodes.append({})

        rewards.append(total_reward)

    env.close()

    # Aggregate per-property violation rates
    per_property_summary = {}
    for pname in PROPERTY_NAMES:
        violations = []
        robustness_vals = []
        for ep_data in per_property_episodes:
            if pname in ep_data and ep_data[pname]["robustness"] is not None:
                robustness_vals.append(ep_data[pname]["robustness"])
                violations.append(not ep_data[pname]["satisfied"])
        if violations:
            per_property_summary[pname] = {
                "violation_rate": float(np.mean(violations)),
                "mean_robustness": float(np.mean(robustness_vals)),
                "std_robustness": float(np.std(robustness_vals)),
            }
        else:
            per_property_summary[pname] = {
                "violation_rate": None,
                "mean_robustness": None,
                "std_robustness": None,
            }

    # Aggregate temporal traces (mean across episodes at each timestep)
    max_len = 0
    for ep_traces in temporal_traces_episodes:
        for pname, trace in ep_traces.items():
            max_len = max(max_len, len(trace))

    temporal_mean = {}
    temporal_std = {}
    for pname in PROPERTY_NAMES:
        all_traces = []
        for ep_traces in temporal_traces_episodes:
            if pname in ep_traces and len(ep_traces[pname]) > 0:
                trace = ep_traces[pname]
                # Pad to max_len with last value
                if len(trace) < max_len:
                    trace = trace + [trace[-1]] * (max_len - len(trace))
                all_traces.append(trace)
        if all_traces:
            arr = np.array(all_traces)
            temporal_mean[pname] = arr.mean(axis=0).tolist()
            temporal_std[pname] = arr.std(axis=0).tolist()
        else:
            temporal_mean[pname] = []
            temporal_std[pname] = []

    return {
        "condition": condition_name,
        "stl_mean_score": float(np.mean(stl_scores)) if stl_scores else None,
        "stl_std_score": float(np.std(stl_scores)) if stl_scores else None,
        "stl_violation_rate": float(np.mean([s < 0 for s in stl_scores])) if stl_scores else None,
        "threshold_violation_rate": float(np.mean(t_vr)) if t_vr else None,
        "cbf_violation_rate": float(np.mean(c_vr)) if c_vr else None,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "per_property": per_property_summary,
        "temporal_mean": temporal_mean,
        "temporal_std": temporal_std,
    }


if __name__ == "__main__":
    # Load trained model
    model_path = None
    for s in [42, 0, 123, 456, 789]:
        p = f"data/checkpoints/seed_{s:03d}/final_model.zip"
        if os.path.exists(p):
            model_path = p
            break
    if model_path is None:
        print("ERROR: No trained model found")
        sys.exit(1)
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path.replace(".zip", ""))

    conditions = [
        ("baseline", {}),
        ("force_5N", {"force_mag": 5.0}),
        ("force_15N", {"force_mag": 15.0}),
        ("force_30N", {"force_mag": 30.0}),
        ("noise_0.01", {"noise_std": 0.01}),
        ("noise_0.05", {"noise_std": 0.05}),
        ("noise_0.15", {"noise_std": 0.15}),
        ("mass_0.8x", {"mass_scale": 0.8}),
        ("mass_1.3x", {"mass_scale": 1.3}),
        ("mass_1.6x", {"mass_scale": 1.6}),
    ]

    all_results = []
    for name, kwargs in conditions:
        t0 = time.time()
        r = run_condition(model, name, n_episodes=5, **kwargs)
        all_results.append(r)
        elapsed = time.time() - t0
        print(f"  {name:12s} | STL={r['stl_mean_score']:+.3f} | Thr={r['threshold_violation_rate']:.0%} | {elapsed:.1f}s")

    # Save full results
    out_path = f"{OUTPUT_DIR}/shift_detailed_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print per-property table
    print(f"\n{'='*100}")
    print("PER-PROPERTY VIOLATION RATES")
    print(f"{'='*100}")
    header = f"{'Condition':12s}" + "".join(f" | {p:>12s}" for p in PROPERTY_NAMES)
    print(header)
    print("-" * 100)
    for r in all_results:
        row = f"{r['condition']:12s}"
        for pname in PROPERTY_NAMES:
            pp = r["per_property"].get(pname, {})
            vr = pp.get("violation_rate")
            if vr is not None:
                row += f" | {vr:>11.0%}"
            else:
                row += f" | {'N/A':>12s}"
        print(row)
