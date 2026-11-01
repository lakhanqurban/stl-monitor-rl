"""Distribution shift / stress testing for safety monitors.

Introduces controlled perturbations at eval time and measures
how STL robustness degrades compared to baselines.
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import json
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from src.environments.locomotion_wrapper import LocomotionWrapper
from src.monitors.stl_monitor import STLSafetyMonitor
from src.monitors.baselines import ThresholdMonitor, CBFMonitor
from src.training.trajectory_logger import TrajectoryLogger


class PerturbedEnv:
    """Wrapper that applies perturbations to a MuJoCo environment.

    Supports: external forces, mass randomization, sensor noise.
    """

    def __init__(
        self,
        env: gym.Env,
        force_magnitude: float = 0.0,
        force_duration: int = 0,
        sensor_noise_std: float = 0.0,
        mass_scale: float = 1.0,
    ):
        self.env = env
        self.force_magnitude = force_magnitude
        self.force_duration = force_duration
        self.sensor_noise_std = sensor_noise_std
        self.mass_scale = mass_scale
        self.force_counter = 0
        self.force_active = False

    def reset(self, seed=None, options=None):
        self.force_counter = 0
        self.force_active = False

        # Apply mass randomization
        if self.mass_scale != 1.0:
            self._apply_mass_change()

        obs, info = self.env.reset(seed=seed, options=options)
        return self._add_noise(obs), info

    def step(self, action):
        # Apply external force periodically
        if self.force_magnitude > 0 and not self.force_active:
            if np.random.random() < 0.05:  # 5% chance per step
                self.force_active = True
                self.force_counter = 0

        if self.force_active:
            self._apply_external_force()
            self.force_counter += 1
            if self.force_counter >= self.force_duration:
                self.force_active = False

        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_noise(obs), reward, terminated, truncated, info

    def _apply_external_force(self):
        """Apply external force to the robot body."""
        data = self.env.unwrapped.data
        if hasattr(data, 'xfrc_applied') and data.xfrc_applied.shape[0] > 0:
            force = np.zeros(6)
            force[:3] = self.force_magnitude * np.random.randn(3)
            data.xfrc_applied[0] = force

    def _apply_mass_change(self):
        """Scale body masses."""
        data = self.env.unwrapped.data
        if hasattr(data, 'body_mass'):
            data.body_mass *= self.mass_scale

    def _add_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to observations."""
        if self.sensor_noise_std > 0:
            noise = np.random.normal(0, self.sensor_noise_std, obs.shape)
            return obs + noise.astype(obs.dtype)
        return obs

    def __getattr__(self, name):
        return getattr(self.env, name)


def test_perturbation_condition(
    model: PPO,
    env_id: str,
    stl_monitor: STLSafetyMonitor,
    threshold_monitor: ThresholdMonitor,
    perturbation_config: dict,
    n_episodes: int = 20,
    seed: int = 0,
) -> dict:
    """Test all monitors under a specific perturbation condition.

    Args:
        model: Trained PPO policy.
        env_id: Base MuJoCo environment ID.
        stl_monitor: STL safety monitor.
        threshold_monitor: Threshold baseline monitor.
        perturbation_config: Dict of perturbation parameters.
        n_episodes: Number of evaluation episodes.
        seed: Random seed.

    Returns:
        Dict with results for each monitor under this condition.
    """
    env = gym.make(env_id, render_mode=None)
    wrapped_env = LocomotionWrapper(env)

    # Apply perturbation
    perturbed_env = PerturbedEnv(
        wrapped_env,
        force_magnitude=perturbation_config.get("force_magnitude", 0.0),
        force_duration=perturbation_config.get("force_duration", 0),
        sensor_noise_std=perturbation_config.get("sensor_noise_std", 0.0),
        mass_scale=perturbation_config.get("mass_scale", 1.0),
    )

    logger = TrajectoryLogger()
    stl_scores = []
    threshold_violations = []
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = perturbed_env.reset(seed=seed + ep)
        logger.reset()
        total_reward = 0.0
        stl_step_results = []

        for t in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = perturbed_env.step(action)
            total_reward += reward

            mon_obs = info.get("monitor_obs", {})
            if mon_obs:
                logger.log_step(t, mon_obs, reward, terminated or truncated)

                # Online STL evaluation
                readings = {k: float(v[0]) if hasattr(v, '__len__') else float(v)
                           for k, v in mon_obs.items()
                           if k in ["body_roll", "body_pitch", "body_height",
                                    "body_velocity_x", "body_velocity_y"]}
                if readings:
                    for prop_name in stl_monitor.monitors:
                        result = stl_monitor.evaluate_online_step(
                            prop_name, t, readings
                        )
                        stl_step_results.append(result)

            if terminated or truncated:
                break

        # Compute episode-level metrics
        signal_data = logger.to_signal_data(
            pd.DataFrame(logger.data) if logger.data["timestep"] else pd.DataFrame()
        ) if logger.data["timestep"] else {}

        if signal_data:
            stl_result = stl_monitor.evaluate_all_trajectory(signal_data)
            stl_score = stl_monitor.get_safety_score(stl_result)
            stl_scores.append(stl_score)

            threshold_result = threshold_monitor.check_trajectory(signal_data)
            threshold_violations.append(threshold_result["violation_rate"])

        episode_rewards.append(total_reward)
        episode_lengths.append(t + 1)

    perturbed_env.close()

    return {
        "perturbation": perturbation_config.get("name", "unknown"),
        "n_episodes": n_episodes,
        "stl": {
            "mean_score": float(np.mean(stl_scores)) if stl_scores else None,
            "std_score": float(np.std(stl_scores)) if stl_scores else None,
            "violation_rate": float(np.mean([s < 0 for s in stl_scores])) if stl_scores else None,
        },
        "threshold": {
            "mean_violation_rate": float(np.mean(threshold_violations)) if threshold_violations else None,
        },
        "episode": {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
        },
    }


def run_distribution_shift_tests(
    model_path: str,
    env_id: str,
    stl_config_path: str,
    perturbation_config_path: str,
    output_dir: str = "data/results",
    n_episodes: int = 20,
    n_seeds: int = 5,
):
    """Run full distribution shift evaluation.

    Tests each perturbation condition across multiple seeds and
    generates the headline comparison figure data.
    """
    import yaml

    # Load configs
    with open(stl_config_path) as f:
        stl_config = yaml.safe_load(f)
    with open(perturbation_config_path) as f:
        pert_config = yaml.safe_load(f)

    # Initialize monitors
    stl_monitor = STLSafetyMonitor(stl_config["properties"])
    threshold_monitor = ThresholdMonitor({
        "body_roll": 0.5,
        "body_pitch": 0.5,
        "body_height": 0.2,
    })

    # Load model
    model = PPO.load(model_path)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Run baseline (no perturbation)
    print("Testing baseline (no perturbation)...")
    baseline_result = test_perturbation_condition(
        model, env_id, stl_monitor, threshold_monitor,
        {"name": "baseline", "force_magnitude": 0.0},
        n_episodes=n_episodes,
    )
    all_results.append(baseline_result)

    # Run each perturbation condition
    for category_name, category in pert_config.get("perturbations", {}).items():
        if not category.get("enabled", False):
            continue

        for variation in category.get("variations", []):
            print(f"Testing {category_name}/{variation['name']}...")
            pert_config_test = {"name": f"{category_name}/{variation['name']}"}
            pert_config_test.update(variation)

            result = test_perturbation_condition(
                model, env_id, stl_monitor, threshold_monitor,
                pert_config_test,
                n_episodes=n_episodes,
            )
            all_results.append(result)

    # Save results
    with open(output_path / "distribution_shift_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate summary table
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            "Condition": r["perturbation"],
            "STL Score (mean±std)": (
                f"{r['stl']['mean_score']:.3f}±{r['stl']['std_score']:.3f}"
                if r["stl"]["mean_score"] is not None else "N/A"
            ),
            "STL Violation Rate": (
                f"{r['stl']['violation_rate']:.1%}"
                if r["stl"]["violation_rate"] is not None else "N/A"
            ),
            "Threshold Violation Rate": (
                f"{r['threshold']['mean_violation_rate']:.1%}"
                if r["threshold"]["mean_violation_rate"] is not None else "N/A"
            ),
            "Mean Reward": f"{r['episode']['mean_reward']:.1f}",
        })

    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path / "shift_summary.csv", index=False)

    print("\nDistribution Shift Results:")
    print(summary_df.to_string(index=False))

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run distribution shift tests")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="Ant-v5")
    parser.add_argument("--stl_config", type=str, default="configs/stl_properties.yaml")
    parser.add_argument("--perturbation_config", type=str, default="configs/perturbation.yaml")
    parser.add_argument("--output_dir", type=str, default="data/results")
    parser.add_argument("--n_episodes", type=int, default=20)
    args = parser.parse_args()

    run_distribution_shift_tests(
        model_path=args.model_path,
        env_id=args.env_id,
        stl_config_path=args.stl_config,
        perturbation_config_path=args.perturbation_config,
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
    )


if __name__ == "__main__":
    main()
