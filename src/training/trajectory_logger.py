"""Trajectory logging utilities for offline STL evaluation.

Logs full trajectories (joint states, body orientation, contact forces)
in parquet format with a defined schema.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


class TrajectoryLogger:
    """Logs trajectory data during RL rollouts for offline analysis.

    Stores observations, actions, rewards, and monitor-relevant signals
    in a structured format (parquet) for repeatable offline STL evaluation.
    """

    def __init__(self, log_dir: str = "data/trajectories"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        """Reset the buffer for a new trajectory."""
        self.data = {
            "timestep": [],
            "body_roll": [],
            "body_pitch": [],
            "body_height": [],
            "body_velocity_x": [],
            "body_velocity_y": [],
            "n_airborne_feet": [],
            "contact_forces_mean": [],
            "reward": [],
            "done": [],
        }

    def log_step(
        self,
        timestep: int,
        monitor_obs: dict,
        reward: float,
        done: bool,
    ):
        """Log a single timestep.

        Args:
            timestep: Current step index.
            monitor_obs: Dict from LocomotionWrapper._extract_monitor_obs().
            reward: Environment reward.
            done: Whether episode ended.
        """
        self.data["timestep"].append(timestep)
        self.data["body_roll"].append(float(monitor_obs["body_roll"][0]))
        self.data["body_pitch"].append(float(monitor_obs["body_pitch"][0]))
        self.data["body_height"].append(float(monitor_obs["body_height"][0]))
        self.data["body_velocity_x"].append(float(monitor_obs["body_velocity_x"][0]))
        self.data["body_velocity_y"].append(float(monitor_obs["body_velocity_y"][0]))
        self.data["n_airborne_feet"].append(int(monitor_obs["n_airborne_feet"][0]))
        self.data["contact_forces_mean"].append(
            float(np.mean(monitor_obs["contact_forces"]))
        )
        self.data["reward"].append(reward)
        self.data["done"].append(done)

    def save(
        self,
        seed: int,
        checkpoint: int,
        episode: int = 0,
    ) -> Path:
        """Save the current trajectory to a parquet file.

        Args:
            seed: Random seed used for this run.
            checkpoint: Training checkpoint number.
            episode: Episode index within the seed/checkpoint.

        Returns:
            Path to the saved parquet file.
        """
        df = pd.DataFrame(self.data)
        filename = f"seed_{seed:03d}_ckpt_{checkpoint:08d}_ep_{episode:04d}.parquet"
        filepath = self.log_dir / filename
        df.to_parquet(filepath, index=False)
        return filepath

    def load(self, filepath: str | Path) -> pd.DataFrame:
        """Load a previously saved trajectory."""
        return pd.read_parquet(filepath)

    def load_all(
        self,
        pattern: str = "*.parquet",
    ) -> list[pd.DataFrame]:
        """Load all trajectories matching a pattern."""
        files = sorted(self.log_dir.glob(pattern))
        return [pd.read_parquet(f) for f in files]

    def to_signal_data(self, df: pd.DataFrame) -> dict[str, list[float]]:
        """Convert a trajectory DataFrame to signal data for RTAMT.

        Returns:
            Dict mapping variable names to lists of float values.
        """
        signal_columns = [
            "body_roll", "body_pitch", "body_height",
            "body_velocity_x", "body_velocity_y",
            "n_airborne_feet", "contact_forces_mean",
        ]
        return {col: df[col].tolist() for col in signal_columns if col in df.columns}


def collect_trajectory(
    env,
    policy,
    max_steps: int = 1000,
    logger: Optional[TrajectoryLogger] = None,
) -> dict:
    """Run a single episode and collect the full trajectory.

    Args:
        env: Wrapped MuJoCo environment (with LocomotionWrapper).
        policy: Trained RL policy (stable-baselines3 style).
        max_steps: Maximum episode length.
        logger: Optional TrajectoryLogger to record to.

    Returns:
        Dict with trajectory data and episode statistics.
    """
    obs, info = env.reset()
    if logger:
        logger.reset()

    total_reward = 0.0
    trajectory = {k: [] for k in [
        "body_roll", "body_pitch", "body_height",
        "body_velocity_x", "body_velocity_y",
        "n_airborne_feet", "contact_forces_mean",
    ]}

    for t in range(max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        mon_obs = info.get("monitor_obs", {})

        for key in trajectory:
            if key in mon_obs:
                val = mon_obs[key]
                trajectory[key].append(float(val[0]) if hasattr(val, '__len__') else float(val))

        if logger:
            logger.log_step(t, mon_obs, reward, terminated or truncated)

        if terminated or truncated:
            break

    return {
        "trajectory": trajectory,
        "total_reward": total_reward,
        "episode_length": t + 1,
    }
