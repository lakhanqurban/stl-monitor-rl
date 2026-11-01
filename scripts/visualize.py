"""Visualize a trained Ant-v5 policy in MuJoCo GUI."""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from stable_baselines3 import PPO
from src.environments.locomotion_wrapper import LocomotionWrapper

# Try seed 456 (highest reward)
model_path = "data/checkpoints/seed_456/final_model.zip"
print(f"Loading model: {model_path}")
model = PPO.load(model_path)

env = LocomotionWrapper(gym.make("Ant-v5", render_mode="human"))
obs, _ = env.reset(seed=42)

total_reward = 0
episodes = 0

for step in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        episodes += 1
        print(f"Episode {episodes}: reward={total_reward:.1f}, steps={step+1}")
        total_reward = 0
        obs, _ = env.reset()

env.close()
print("Done.")
