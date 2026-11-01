"""Check HalfCheetah pitch distribution and run distribution shift experiments."""
import os, warnings, sys, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environments.locomotion_wrapper import LocomotionWrapper, ENV_CONFIGS
from src.monitors.stl_monitor import STLSafetyMonitor
from src.monitors.baselines import ThresholdMonitor, CBFMonitor
import yaml

ENV_ID = "HalfCheetah-v5"
OUTPUT_DIR = "data/checkpoints_halfcheetah"

with open("configs/stl_properties_halfcheetah.yaml") as f:
    stl_config = yaml.safe_load(f)

# Load best model (seed 123 has highest reward)
model = PPO.load(f"{OUTPUT_DIR}/seed_123/final_model")
env = DummyVecEnv([lambda: LocomotionWrapper(gym.make(ENV_ID, render_mode=None), env_name=ENV_ID)])
env = VecNormalize.load(f"{OUTPUT_DIR}/seed_123/vec_normalize.pkl", env)
env.training = False

# Collect pitch data from nominal run
all_pitches = []
all_heights = []
all_velocities = []

for ep in range(10):
    obs = env.reset()
    for t in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        mon_obs = info[0].get("monitor_obs", {})
        if mon_obs:
            all_pitches.append(float(mon_obs["body_pitch"][0]))
            all_heights.append(float(mon_obs["body_height"][0]))
            all_velocities.append(float(mon_obs["body_velocity_x"][0]))
        if done[0]:
            break

env.close()

print("HalfCheetah Nominal Statistics:")
print(f"  Pitch: mean={np.mean(all_pitches):.4f}, std={np.std(all_pitches):.4f}, min={np.min(all_pitches):.4f}, max={np.max(all_pitches):.4f}")
print(f"  Pitch P5={np.percentile(all_pitches, 5):.4f}, P95={np.percentile(all_pitches, 95):.4f}")
print(f"  Height: mean={np.mean(all_heights):.4f}, std={np.std(all_heights):.4f}, min={np.min(all_heights):.4f}, max={np.max(all_heights):.4f}")
print(f"  Height P5={np.percentile(all_heights, 5):.4f}, P95={np.percentile(all_heights, 95):.4f}")
print(f"  Velocity: mean={np.mean(all_velocities):.4f}, std={np.std(all_velocities):.4f}, min={np.min(all_velocities):.4f}, max={np.max(all_velocities):.4f}")
print(f"  Velocity P5={np.percentile(all_velocities, 5):.4f}, P95={np.percentile(all_velocities, 95):.4f}")

# Check how many steps violate different thresholds
for pitch_thresh in [0.3, 0.5, 0.8, 1.0]:
    violations = np.mean(np.abs(all_pitches) > pitch_thresh)
    print(f"  Pitch > {pitch_thresh}: {violations:.1%} violation")

for height_thresh in [0.2, 0.3, 0.4]:
    violations = np.mean(np.array(all_heights) < height_thresh)
    print(f"  Height < {height_thresh}: {violations:.1%} violation")
