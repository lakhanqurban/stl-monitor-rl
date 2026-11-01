"""Distribution shift experiments for HalfCheetah-v5."""
import os, warnings, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
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
from src.monitors.baselines import ThresholdMonitor, CBFMonitor
import yaml

ENV_ID = "HalfCheetah-v5"
OUTPUT_DIR = PROJECT_ROOT / "data" / "checkpoints_halfcheetah"
RESULTS_DIR = PROJECT_ROOT / "data" / "results_halfcheetah"

with open("configs/stl_properties_halfcheetah.yaml") as f:
    stl_config = yaml.safe_load(f)

os.makedirs(RESULTS_DIR, exist_ok=True)

model = PPO.load(f"{OUTPUT_DIR}/seed_123/final_model.zip")


class PerturbedHalfCheetah:
    def __init__(self, env, force_magnitude=0.0, sensor_noise_std=0.0, mass_scale=1.0):
        self.env = env
        self.force_magnitude = force_magnitude
        self.sensor_noise_std = sensor_noise_std
        self.mass_scale = mass_scale

    def reset(self, seed=None, options=None):
        # Apply mass BEFORE reset so it persists
        if self.mass_scale != 1.0:
            model = self.env.unwrapped.model
            if not hasattr(self, '_orig_mass'):
                self._orig_mass = model.body_mass.copy()
            model.body_mass[:] = self._orig_mass * self.mass_scale

        obs, info = self.env.reset(seed=seed, options=options)
        return self._add_noise(obs), info

    def step(self, action):
        if self.force_magnitude > 0:
            data = self.env.unwrapped.data
            if hasattr(data, 'xfrc_applied') and data.xfrc_applied.shape[0] > 1:
                force = np.zeros(6)
                force[:3] = self.force_magnitude * np.random.randn(3)
                data.xfrc_applied[1] = force
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_noise(obs), reward, terminated, truncated, info

    def _add_noise(self, obs):
        if self.sensor_noise_std > 0:
            return obs + np.random.normal(0, self.sensor_noise_std, obs.shape).astype(obs.dtype)
        return obs

    def close(self):
        self.env.close()


def evaluate_condition(env, model, stl_monitor, tm, cbf, n_episodes=5, max_steps=500):
    stl_scores = []
    threshold_vrs = []
    cbf_vrs = []
    rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep * 100)
        logger = TrajectoryLogger()
        total_reward = 0

        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            mon_obs = info.get("monitor_obs", {})
            if mon_obs:
                logger.log_step(t, mon_obs, reward, terminated or truncated)

            if terminated or truncated:
                break

        signal_data = logger.to_signal_data(pd.DataFrame(logger.data)) if logger.data.get("timestep") else {}
        if signal_data:
            stl_result = stl_monitor.evaluate_all_trajectory(signal_data)
            stl_scores.append(stl_monitor.get_safety_score(stl_result))
            threshold_vrs.append(tm.check_trajectory(signal_data)["violation_rate"])
            cbf_vrs.append(cbf.check_trajectory(signal_data)["violation_rate"])
        rewards.append(total_reward)

    return {
        "stl_mean": float(np.mean(stl_scores)) if stl_scores else 0.0,
        "stl_std": float(np.std(stl_scores)) if stl_scores else 0.0,
        "threshold_vr": float(np.mean(threshold_vrs)) if threshold_vrs else 0.0,
        "cbf_vr": float(np.mean(cbf_vrs)) if cbf_vrs else 0.0,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
    }


stl_monitor = STLSafetyMonitor(stl_config["properties"])
tm = ThresholdMonitor(
    {"body_pitch": 0.8, "body_height": 0.12},
    {"body_pitch": "both", "body_height": "lower"},
)
cbf = CBFMonitor(3, {"body_pitch": (-0.8, 0.8), "body_height": (0.12, 5.0)})

conditions = [
    {"name": "baseline", "force_magnitude": 0.0, "sensor_noise_std": 0.0, "mass_scale": 1.0},
    {"name": "force_5N", "force_magnitude": 5.0, "sensor_noise_std": 0.0, "mass_scale": 1.0},
    {"name": "force_15N", "force_magnitude": 15.0, "sensor_noise_std": 0.0, "mass_scale": 1.0},
    {"name": "noise_0.01", "force_magnitude": 0.0, "sensor_noise_std": 0.01, "mass_scale": 1.0},
    {"name": "noise_0.05", "force_magnitude": 0.0, "sensor_noise_std": 0.05, "mass_scale": 1.0},
    {"name": "noise_0.15", "force_magnitude": 0.0, "sensor_noise_std": 0.15, "mass_scale": 1.0},
    {"name": "mass_0.8x", "force_magnitude": 0.0, "sensor_noise_std": 0.0, "mass_scale": 0.8},
    {"name": "mass_1.3x", "force_magnitude": 0.0, "sensor_noise_std": 0.0, "mass_scale": 1.3},
    {"name": "mass_1.6x", "force_magnitude": 0.0, "sensor_noise_std": 0.0, "mass_scale": 1.6},
]

all_results = []
for cond in conditions:
    print(f"Testing {cond['name']}...")
    raw_env = gym.make(ENV_ID, render_mode=None)
    wrapped = LocomotionWrapper(raw_env, env_name=ENV_ID)
    perturbed = PerturbedHalfCheetah(
        wrapped,
        force_magnitude=cond["force_magnitude"],
        sensor_noise_std=cond["sensor_noise_std"],
        mass_scale=cond["mass_scale"],
    )
    result = evaluate_condition(perturbed, model, stl_monitor, tm, cbf)
    result["condition"] = cond["name"]
    all_results.append(result)
    print(f"  STL={result['stl_mean']:.4f}, threshold_VR={result['threshold_vr']:.1%}, reward={result['reward_mean']:.1f}")
    raw_env.close()

with open(f"{RESULTS_DIR}/shift_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nDone!")
