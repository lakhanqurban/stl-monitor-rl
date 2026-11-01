"""Phase 4: Precision/Recall/FPR comparison table for all monitors.

Evaluates STL, Threshold, AnomalyDetector, and CBF on nominal vs.
perturbed trajectories to compute detection quality metrics.
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
from src.training.trajectory_logger import TrajectoryLogger, collect_trajectory
from src.monitors.stl_monitor import STLSafetyMonitor
from src.monitors.baselines import ThresholdMonitor, AnomalyDetector, CBFMonitor
import yaml

ENV_ID = "Ant-v5"
OUTPUT_DIR = "data/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("configs/stl_properties.yaml") as f:
    stl_config = yaml.safe_load(f)


def collect_episodes(model, n_episodes=10, max_steps=500):
    """Collect nominal and perturbed episodes."""
    nominal_episodes = []
    perturbed_episodes = []

    raw_env = gym.make(ENV_ID, render_mode=None)
    wrapped = LocomotionWrapper(raw_env)

    for ep in range(n_episodes):
        # Nominal
        logger = TrajectoryLogger()
        result = collect_trajectory(wrapped, model, max_steps=max_steps, logger=logger)
        df = pd.DataFrame(logger.data) if logger.data["timestep"] else pd.DataFrame()
        if not df.empty:
            nominal_episodes.append({"df": df, "signal": logger.to_signal_data(df), "reward": result["total_reward"]})

        # Perturbed (sensor noise sigma=0.15)
        obs, info = wrapped.reset(seed=ep * 100 + 1)
        total_reward = 0
        logger_p = TrajectoryLogger()
        for t in range(max_steps):
            noisy_obs = obs + np.random.normal(0, 0.15, obs.shape).astype(obs.dtype)
            action, _ = model.predict(noisy_obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            total_reward += reward
            mon_obs = info.get("monitor_obs", {})
            logger_p.log_step(t, mon_obs, reward, terminated or truncated)
            if terminated or truncated:
                break
        df_p = pd.DataFrame(logger_p.data) if logger_p.data["timestep"] else pd.DataFrame()
        if not df_p.empty:
            perturbed_episodes.append({"df": df_p, "signal": logger_p.to_signal_data(df_p), "reward": total_reward})

    raw_env.close()
    return nominal_episodes, perturbed_episodes


def evaluate_monitors(nominal_eps, perturbed_eps, stl_monitor):
    """Compute precision/recall/FPR for each monitor.

    Ground truth: nominal = safe (negative), perturbed = unsafe (positive).
    """
    tm = ThresholdMonitor(
        {"body_roll": 1.0, "body_pitch": 1.0, "body_height": 0.4},
        {"body_roll": "both", "body_pitch": "both", "body_height": "lower"},
    )
    cbf = CBFMonitor(6, {"body_roll": (-1.0, 1.0), "body_pitch": (-1.0, 1.0), "body_height": (0.4, 5.0)})

    # Train anomaly detector on nominal data
    nominal_mon_obs = []
    for ep in nominal_eps:
        sig = ep["signal"]
        obs_array = np.column_stack([sig[k] for k in sorted(sig.keys())])
        nominal_mon_obs.append(obs_array)

    if nominal_mon_obs:
        input_dim = nominal_mon_obs[0].shape[1]
    else:
        input_dim = 7
    ad = AnomalyDetector(input_dim=input_dim, hidden_dim=16, threshold_percentile=95.0)
    if nominal_mon_obs:
        all_nominal = np.vstack(nominal_mon_obs)
        ad.fit(all_nominal, epochs=30)

    results = {"stl": [], "threshold": [], "anomaly": [], "cbf": []}

    # Evaluate nominal episodes (label=0, safe)
    for ep in nominal_eps:
        sig = ep["signal"]

        # STL
        stl_res = stl_monitor.evaluate_all_trajectory(sig)
        stl_score = stl_monitor.get_safety_score(stl_res)
        results["stl"].append({"label": 0, "detected": stl_score < 0, "score": stl_score})

        # Threshold
        thr_res = tm.check_trajectory(sig)
        results["threshold"].append({"label": 0, "detected": thr_res["violation_rate"] > 0, "score": thr_res["violation_rate"]})

        # Anomaly detector
        obs_array = np.column_stack([sig[k] for k in sorted(sig.keys())])
        ad_res = ad.check_trajectory(obs_array)
        results["anomaly"].append({"label": 0, "detected": ad_res["anomaly_rate"] > 0.1, "score": ad_res["anomaly_rate"]})

        # CBF
        cbf_res = cbf.check_trajectory(sig)
        results["cbf"].append({"label": 0, "detected": cbf_res["violation_rate"] > 0, "score": cbf_res["violation_rate"]})

    # Evaluate perturbed episodes (label=1, unsafe)
    for ep in perturbed_eps:
        sig = ep["signal"]

        stl_res = stl_monitor.evaluate_all_trajectory(sig)
        stl_score = stl_monitor.get_safety_score(stl_res)
        results["stl"].append({"label": 1, "detected": stl_score < 0, "score": stl_score})

        thr_res = tm.check_trajectory(sig)
        results["threshold"].append({"label": 1, "detected": thr_res["violation_rate"] > 0, "score": thr_res["violation_rate"]})

        obs_array = np.column_stack([sig[k] for k in sorted(sig.keys())])
        ad_res = ad.check_trajectory(obs_array)
        results["anomaly"].append({"label": 1, "detected": ad_res["anomaly_rate"] > 0.1, "score": ad_res["anomaly_rate"]})

        cbf_res = cbf.check_trajectory(sig)
        results["cbf"].append({"label": 1, "detected": cbf_res["violation_rate"] > 0, "score": cbf_res["violation_rate"]})

    return results


def compute_metrics(results):
    """Compute precision, recall, FPR, F1 for each monitor."""
    table = []
    for monitor_name, preds in results.items():
        labels = [p["label"] for p in preds]
        detected = [p["detected"] for p in preds]

        tp = sum(1 for l, d in zip(labels, detected) if l == 1 and d)
        fp = sum(1 for l, d in zip(labels, detected) if l == 0 and d)
        tn = sum(1 for l, d in zip(labels, detected) if l == 0 and not d)
        fn = sum(1 for l, d in zip(labels, detected) if l == 1 and not d)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        table.append({
            "Monitor": monitor_name,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Precision": f"{precision:.3f}",
            "Recall": f"{recall:.3f}",
            "FPR": f"{fpr:.3f}",
            "F1": f"{f1:.3f}",
        })
    return pd.DataFrame(table)


if __name__ == "__main__":
    # Load model
    model_path = None
    for s in [42, 0, 123, 456, 789]:
        p = f"data/checkpoints/seed_{s:03d}/final_model.zip"
        if os.path.exists(p):
            model_path = p
            break
    if not model_path:
        print("No trained model found. Train first.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    stl_monitor = STLSafetyMonitor(stl_config["properties"])

    print("Collecting nominal + perturbed episodes...")
    nominal, perturbed = collect_episodes(model, n_episodes=10, max_steps=500)
    print(f"  Nominal: {len(nominal)}, Perturbed: {len(perturbed)}")

    print("Evaluating monitors...")
    results = evaluate_monitors(nominal, perturbed, stl_monitor)

    metrics_df = compute_metrics(results)
    metrics_df.to_csv(f"{OUTPUT_DIR}/detection_metrics.csv", index=False)

    print("\n" + "=" * 70)
    print("DETECTION QUALITY METRICS (Nominal=Safe, Perturbed=Unsafe)")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}/detection_metrics.csv")
