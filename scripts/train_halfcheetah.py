"""Train and evaluate HalfCheetah-v5 with STL monitoring.

5 seeds x 200k steps, collect trajectories, evaluate with STL monitor.
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environments.locomotion_wrapper import LocomotionWrapper
from src.training.trajectory_logger import TrajectoryLogger, collect_trajectory
from src.monitors.stl_monitor import STLSafetyMonitor
from src.monitors.baselines import ThresholdMonitor, CBFMonitor
import yaml

SEEDS = [0, 42, 123, 456, 789]
TOTAL_TIMESTEPS = 200_000
ENV_ID = "HalfCheetah-v5"
OUTPUT_DIR = "data/checkpoints_halfcheetah"
TRAJ_DIR = "data/trajectories_halfcheetah"

# Use HalfCheetah-specific STL properties
with open("configs/stl_properties_halfcheetah.yaml") as f:
    stl_config = yaml.safe_load(f)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)


def train_seed(seed, total_timesteps):
    print(f"\nTraining seed={seed}, timesteps={total_timesteps}")
    t0 = time.time()

    env = DummyVecEnv([lambda: LocomotionWrapper(gym.make(ENV_ID, render_mode=None), env_name=ENV_ID)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        verbose=0, seed=seed,
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    model.save(f"{OUTPUT_DIR}/seed_{seed:03d}/final_model")
    env.save(f"{OUTPUT_DIR}/seed_{seed:03d}/vec_normalize.pkl")
    env.close()

    print(f"  Done in {time.time()-t0:.1f}s")
    return model


def collect_and_evaluate(model, seed):
    raw_env = gym.make(ENV_ID, render_mode=None)
    wrapped = LocomotionWrapper(raw_env, env_name=ENV_ID)
    logger = TrajectoryLogger(TRAJ_DIR)

    all_trajectories = []
    for ep in range(5):
        logger.reset()
        result = collect_trajectory(wrapped, model, max_steps=500, logger=logger)
        filepath = logger.save(seed=seed, checkpoint=0, episode=ep)
        all_trajectories.append({"file": str(filepath), "reward": result["total_reward"], "length": result["episode_length"]})
    raw_env.close()

    monitor = STLSafetyMonitor(stl_config["properties"])
    stl_scores = []
    for traj_data in all_trajectories:
        df = logger.load(traj_data["file"])
        signal_data = logger.to_signal_data(df)
        results = monitor.evaluate_all_trajectory(signal_data)
        stl_scores.append(monitor.get_safety_score(results))

    tm = ThresholdMonitor(
        {"body_pitch": 0.5, "body_height": 0.3},
        {"body_pitch": "both", "body_height": "lower"},
    )
    threshold_vr = []
    cbf = CBFMonitor(3, {"body_pitch": (-0.5, 0.5), "body_height": (0.3, 5.0)})
    cbf_vr = []
    for traj_data in all_trajectories:
        df = logger.load(traj_data["file"])
        sd = logger.to_signal_data(df)
        threshold_vr.append(tm.check_trajectory(sd)["violation_rate"])
        cbf_vr.append(cbf.check_trajectory(sd)["violation_rate"])

    return {
        "seed": seed,
        "env": ENV_ID,
        "mean_reward": float(np.mean([t["reward"] for t in all_trajectories])),
        "stl_mean_score": float(np.mean(stl_scores)),
        "stl_std_score": float(np.std(stl_scores)),
        "stl_violation_rate": float(np.mean([s < 0 for s in stl_scores])),
        "threshold_violation_rate": float(np.mean(threshold_vr)),
        "cbf_violation_rate": float(np.mean(cbf_vr)),
    }


if __name__ == "__main__":
    all_results = []
    for seed in SEEDS:
        model = train_seed(seed, TOTAL_TIMESTEPS)
        eval_result = collect_and_evaluate(model, seed)
        all_results.append(eval_result)
        with open(f"{OUTPUT_DIR}/results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Seed {seed}: reward={eval_result['mean_reward']:.1f}, STL={eval_result['stl_mean_score']:.4f}")

    print(f"\n{'='*50}")
    print("SUMMARY (mean +/- std across 5 seeds)")
    print(f"{'='*50}")
    for key in ["mean_reward", "stl_mean_score", "stl_violation_rate", "threshold_violation_rate", "cbf_violation_rate"]:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
