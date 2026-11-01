"""Training script for baseline RL policies using stable-baselines3.

Trains PPO on MuJoCo locomotion tasks with multiple seeds,
saves checkpoints, and logs trajectories for offline STL evaluation.
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environments.locomotion_wrapper import LocomotionWrapper
from src.training.trajectory_logger import TrajectoryLogger, collect_trajectory


def make_env(env_id: str, max_episode_steps: int = 1000):
    """Create a wrapped MuJoCo environment."""
    def _init():
        env = gym.make(env_id, render_mode=None)
        env = LocomotionWrapper(env, max_episode_steps=max_episode_steps)
        return env
    return _init


def train_single_seed(
    env_id: str,
    seed: int,
    total_timesteps: int,
    output_dir: str,
    n_eval_episodes: int = 10,
):
    """Train PPO for a single seed and save results.

    Args:
        env_id: MuJoCo environment ID (e.g., "Ant-v5").
        seed: Random seed.
        total_timesteps: Total training timesteps.
        output_dir: Directory to save checkpoints and logs.
        n_eval_episodes: Number of episodes for evaluation.
    """
    output_path = Path(output_dir) / f"seed_{seed:03d}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Create vectorized environments
    env = DummyVecEnv([make_env(env_id)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(env_id)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        tensorboard_log=str(output_path / "tensorboard"),
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(output_path / "checkpoints"),
        name_prefix="ppo_loc",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / "best_model"),
        log_path=str(output_path / "eval"),
        eval_freq=50_000,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model and normalization stats
    model.save(str(output_path / "final_model"))
    env.save(str(output_path / "vec_normalize.pkl"))

    env.close()
    eval_env.close()

    return model, output_path


def collect_trajectories(
    env_id: str,
    model_path: str,
    n_episodes: int = 20,
    seed: int = 0,
    log_dir: str = "data/trajectories",
) -> list[dict]:
    """Collect trajectories from a trained policy for offline STL evaluation.

    Args:
        env_id: MuJoCo environment ID.
        model_path: Path to saved PPO model.
        n_episodes: Number of episodes to collect.
        seed: Random seed.
        log_dir: Directory to save trajectory logs.

    Returns:
        List of trajectory dicts.
    """
    env = gym.make(env_id, render_mode=None)
    env = LocomotionWrapper(env)

    model = PPO.load(model_path)
    logger = TrajectoryLogger(log_dir)

    trajectories = []
    for ep in range(n_episodes):
        logger.reset()
        result = collect_trajectory(env, model, logger=logger)

        filepath = logger.save(seed=seed, checkpoint=0, episode=ep)
        trajectories.append({
            "file": str(filepath),
            "total_reward": result["total_reward"],
            "episode_length": result["episode_length"],
        })

    env.close()
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Train baseline RL policy")
    parser.add_argument("--env_id", type=str, default="Ant-v5", help="MuJoCo env ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--output_dir", type=str, default="data/checkpoints")
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    args = parser.parse_args()

    print(f"Training {args.env_id} with seed {args.seed} for {args.total_timesteps} steps")
    model, output_path = train_single_seed(
        env_id=args.env_id,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        output_dir=args.output_dir,
        n_eval_episodes=args.n_eval_episodes,
    )

    # Collect trajectories from the trained model
    print("Collecting trajectories for offline evaluation...")
    trajectories = collect_trajectories(
        env_id=args.env_id,
        model_path=str(output_path / "final_model"),
        n_episodes=20,
        seed=args.seed,
    )

    # Save metadata
    metadata = {
        "env_id": args.env_id,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "trajectories": trajectories,
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
