"""Phase 8: Ablation studies.

1. Drop each STL property individually -> detection quality degradation
2. Vary time-bound parameters (T, theta, h_min) -> sensitivity analysis
3. Single-seed vs. multi-seed variance illustration
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import sys, json, time, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.environments.locomotion_wrapper import LocomotionWrapper
from src.training.trajectory_logger import collect_trajectory, TrajectoryLogger
from src.monitors.stl_monitor import STLSafetyMonitor
import yaml

ENV_ID = "Ant-v5"
OUTPUT_DIR = "data/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("configs/stl_properties.yaml") as f:
    full_stl_config = yaml.safe_load(f)


def collect_nominal_episodes(model, n_episodes=10, max_steps=500):
    """Collect nominal (no perturbation) episodes."""
    raw_env = gym.make(ENV_ID, render_mode=None)
    wrapped = LocomotionWrapper(raw_env)
    episodes = []
    for ep in range(n_episodes):
        logger = TrajectoryLogger()
        result = collect_trajectory(wrapped, model, max_steps=max_steps, logger=logger)
        df = pd.DataFrame(logger.data) if logger.data["timestep"] else pd.DataFrame()
        if not df.empty:
            episodes.append(logger.to_signal_data(df))
    raw_env.close()
    return episodes


def collect_perturbed_episodes(model, noise_std=0.15, n_episodes=10, max_steps=500):
    """Collect perturbed (sensor noise) episodes."""
    raw_env = gym.make(ENV_ID, render_mode=None)
    wrapped = LocomotionWrapper(raw_env)
    episodes = []
    for ep in range(n_episodes):
        obs, info = wrapped.reset(seed=ep * 100 + 1)
        logger = TrajectoryLogger()
        for t in range(max_steps):
            noisy_obs = obs + np.random.normal(0, noise_std, obs.shape).astype(obs.dtype)
            action, _ = model.predict(noisy_obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            mon_obs = info.get("monitor_obs", {})
            logger.log_step(t, mon_obs, reward, terminated or truncated)
            if terminated or truncated:
                break
        df = pd.DataFrame(logger.data) if logger.data["timestep"] else pd.DataFrame()
        if not df.empty:
            episodes.append(logger.to_signal_data(df))
    raw_env.close()
    return episodes


def compute_detection_stats(nominal_scores, perturbed_scores):
    """Compute detection rate from robustness scores."""
    nom_violations = sum(1 for s in nominal_scores if s < 0)
    pert_violations = sum(1 for s in perturbed_scores if s < 0)
    n_nom = len(nominal_scores)
    n_pert = len(perturbed_scores)
    return {
        "nominal_violation_rate": nom_violations / n_nom if n_nom > 0 else 0,
        "perturbed_violation_rate": pert_violations / n_pert if n_pert > 0 else 0,
        "detection_rate": pert_violations / n_pert if n_pert > 0 else 0,
        "false_positive_rate": nom_violations / n_nom if n_nom > 0 else 0,
        "nominal_scores": nominal_scores,
        "perturbed_scores": perturbed_scores,
    }


def ablation_property_drop(nominal_eps, perturbed_eps):
    """Ablation 1: Drop each property and measure detection degradation."""
    all_props = list(full_stl_config["properties"].keys())
    results = []

    # Full config baseline
    full_monitor = STLSafetyMonitor(full_stl_config["properties"])
    full_nom = [full_monitor.get_safety_score(full_monitor.evaluate_all_trajectory(ep)) for ep in nominal_eps]
    full_pert = [full_monitor.get_safety_score(full_monitor.evaluate_all_trajectory(ep)) for ep in perturbed_eps]
    full_stats = compute_detection_stats(full_nom, full_pert)
    results.append({"dropped": "none (full)", **{k: v for k, v in full_stats.items() if k not in ["nominal_scores", "perturbed_scores"]}})

    # Drop each property
    for prop_to_drop in all_props:
        reduced_props = {k: v for k, v in full_stl_config["properties"].items() if k != prop_to_drop}
        monitor = STLSafetyMonitor(reduced_props)
        nom_scores = [monitor.get_safety_score(monitor.evaluate_all_trajectory(ep)) for ep in nominal_eps]
        pert_scores = [monitor.get_safety_score(monitor.evaluate_all_trajectory(ep)) for ep in perturbed_eps]
        stats = compute_detection_stats(nom_scores, pert_scores)
        results.append({
            "dropped": prop_to_drop,
            **{k: v for k, v in stats.items() if k not in ["nominal_scores", "perturbed_scores"]},
        })

    return results


def ablation_parameter_sweep(nominal_eps, perturbed_eps):
    """Ablation 2: Sweep key parameters and measure sensitivity."""
    results = []

    # Sweep height threshold h_min
    for h_min in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
        config = json.loads(json.dumps(full_stl_config["properties"]))
        config["no_falls"]["parameters"]["h_min"] = h_min
        config["no_falls"]["formula"] = f"G[0,1000](height > {{{{h_min}}}})".replace("{{h_min}}", "{h_min}")
        monitor = STLSafetyMonitor(config)
        nom_scores = [monitor.get_safety_score(monitor.evaluate_all_trajectory(ep)) for ep in nominal_eps]
        pert_scores = [monitor.get_safety_score(monitor.evaluate_all_trajectory(ep)) for ep in perturbed_eps]
        stats = compute_detection_stats(nom_scores, pert_scores)
        results.append({
            "sweep": "h_min",
            "value": h_min,
            **{k: v for k, v in stats.items() if k not in ["nominal_scores", "perturbed_scores"]},
        })

    # Roll threshold theta
    for theta in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        config = json.loads(json.dumps(full_stl_config["properties"]))
        config["stability_roll"]["parameters"]["neg_threshold"] = -theta
        config["stability_roll"]["parameters"]["pos_threshold"] = theta
        monitor = STLSafetyMonitor(config)
        nom_scores = [monitor.get_safety_score(monitor.evaluate_all_trajectory(ep)) for ep in nominal_eps]
        pert_scores = [monitor.get_safety_score(monitor.evaluate_all_trajectory(ep)) for ep in perturbed_eps]
        stats = compute_detection_stats(nom_scores, pert_scores)
        results.append({
            "sweep": "roll_threshold",
            "value": theta,
            **{k: v for k, v in stats.items() if k not in ["nominal_scores", "perturbed_scores"]},
        })

    return results


def ablation_seed_variance(model_paths, nominal_eps_per_seed=5, max_steps=500):
    """Ablation 3: Multi-seed variance illustration."""
    monitor = STLSafetyMonitor(full_stl_config["properties"])
    seed_results = []

    for seed, path in model_paths.items():
        model = PPO.load(path)
        eps = collect_nominal_episodes(model, n_episodes=nominal_eps_per_seed, max_steps=max_steps)
        scores = [monitor.get_safety_score(monitor.evaluate_all_trajectory(ep)) for ep in eps]
        seed_results.append({
            "seed": seed,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "violation_rate": float(np.mean([s < 0 for s in scores])),
            "scores": scores,
        })

    return seed_results


if __name__ == "__main__":
    # Find available models
    model_paths = {}
    for s in [0, 42, 123, 456, 789]:
        p = f"data/checkpoints/seed_{s:03d}/final_model.zip"
        if os.path.exists(p):
            model_paths[s] = p

    if not model_paths:
        print("No trained models found.")
        sys.exit(1)

    primary_seed = list(model_paths.keys())[0]
    model = PPO.load(model_paths[primary_seed])
    print(f"Using primary model: seed {primary_seed}")

    print("Collecting nominal episodes...")
    nominal_eps = collect_nominal_episodes(model, n_episodes=10, max_steps=500)
    print(f"  {len(nominal_eps)} nominal episodes")

    print("Collecting perturbed episodes (noise=0.15)...")
    perturbed_eps = collect_perturbed_episodes(model, noise_std=0.15, n_episodes=10, max_steps=500)
    print(f"  {len(perturbed_eps)} perturbed episodes")

    # Ablation 1: Property drop
    print("\n--- Ablation 1: Property Drop ---")
    drop_results = ablation_property_drop(nominal_eps, perturbed_eps)
    for r in drop_results:
        print(f"  Dropped: {r['dropped']:25s} | Det: {r['detection_rate']:.0%} | FPR: {r['false_positive_rate']:.0%}")

    # Ablation 2: Parameter sweep
    print("\n--- Ablation 2: Parameter Sweep ---")
    sweep_results = ablation_parameter_sweep(nominal_eps, perturbed_eps)
    for r in sweep_results:
        print(f"  {r['sweep']:15s} = {r['value']:5.2f} | Det: {r['detection_rate']:.0%} | FPR: {r['false_positive_rate']:.0%}")

    # Ablation 3: Seed variance
    print("\n--- Ablation 3: Seed Variance ---")
    seed_results = ablation_seed_variance(model_paths, nominal_eps_per_seed=5, max_steps=500)
    for r in seed_results:
        print(f"  Seed {r['seed']:3d} | Score: {r['mean_score']:+.3f} +/- {r['std_score']:.3f} | Viol: {r['violation_rate']:.0%}")

    # Save all results
    all_results = {
        "property_drop": [{k: v for k, v in r.items()} for r in drop_results],
        "parameter_sweep": [{k: v for k, v in r.items()} for r in sweep_results],
        "seed_variance": [{k: v for k, v in r.items() if k != "scores"} for r in seed_results],
    }
    with open(f"{OUTPUT_DIR}/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR}/ablation_results.json")
