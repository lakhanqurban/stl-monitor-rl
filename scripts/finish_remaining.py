"""Complete remaining seeds (456, 789)."""
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from src.environments.locomotion_wrapper import LocomotionWrapper
from src.training.trajectory_logger import TrajectoryLogger, collect_trajectory
from src.monitors.stl_monitor import STLSafetyMonitor
from src.monitors.baselines import ThresholdMonitor, CBFMonitor
import yaml

REMAINING_SEEDS = [456, 789]
TOTAL_TIMESTEPS = 200_000
ENV_ID = "Ant-v5"
OUTPUT_DIR = "data/checkpoints"
TRAJ_DIR = "data/trajectories"

with open("configs/stl_properties.yaml") as f:
    stl_config = yaml.safe_load(f)


def train_and_eval(seed):
    print(f"Seed {seed}: training...")
    t0 = time.time()

    env = DummyVecEnv([lambda: LocomotionWrapper(gym.make(ENV_ID, render_mode=None))])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, verbose=0, seed=seed)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=False)
    model.save(f"{OUTPUT_DIR}/seed_{seed:03d}/final_model")
    env.save(f"{OUTPUT_DIR}/seed_{seed:03d}/vec_normalize.pkl")
    env.close()
    print(f"  Trained in {time.time()-t0:.1f}s")

    raw_env = gym.make(ENV_ID, render_mode=None)
    wrapped = LocomotionWrapper(raw_env)
    logger = TrajectoryLogger(TRAJ_DIR)
    trajs = []
    for ep in range(5):
        logger.reset()
        result = collect_trajectory(wrapped, model, max_steps=500, logger=logger)
        fp = logger.save(seed=seed, checkpoint=0, episode=ep)
        trajs.append({"file": str(fp), "reward": result["total_reward"], "length": result["episode_length"]})
    raw_env.close()

    monitor = STLSafetyMonitor(stl_config["properties"])
    stl_scores = []
    tm = ThresholdMonitor(
        {"body_roll": 1.0, "body_pitch": 1.0, "body_height": 0.4},
        {"body_roll": "both", "body_pitch": "both", "body_height": "lower"},
    )
    cbf = CBFMonitor(6, {"body_roll": (-1.0, 1.0), "body_pitch": (-1.0, 1.0), "body_height": (0.4, 5.0)})
    t_vr, c_vr = [], []
    for td in trajs:
        df = logger.load(td["file"])
        sd = logger.to_signal_data(df)
        stl_scores.append(monitor.get_safety_score(monitor.evaluate_all_trajectory(sd)))
        t_vr.append(tm.check_trajectory(sd)["violation_rate"])
        c_vr.append(cbf.check_trajectory(sd)["violation_rate"])

    return {
        "seed": seed,
        "mean_reward": float(np.mean([t["reward"] for t in trajs])),
        "stl_mean_score": float(np.mean(stl_scores)),
        "stl_std_score": float(np.std(stl_scores)),
        "stl_violation_rate": float(np.mean([s < 0 for s in stl_scores])),
        "threshold_violation_rate": float(np.mean(t_vr)),
        "cbf_violation_rate": float(np.mean(c_vr)),
    }


if __name__ == "__main__":
    with open(f"{OUTPUT_DIR}/results.json") as f:
        all_results = json.load(f)

    for seed in REMAINING_SEEDS:
        r = train_and_eval(seed)
        all_results.append(r)
        with open(f"{OUTPUT_DIR}/results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Seed {seed}: reward={r['mean_reward']:.1f}, STL={r['stl_mean_score']:.4f}")

    print(f"\nAll 5 seeds complete!")
    for key in ["mean_reward", "stl_mean_score", "stl_violation_rate", "threshold_violation_rate", "cbf_violation_rate"]:
        vals = [r[key] for r in all_results]
        print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
