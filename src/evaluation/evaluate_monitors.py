"""Evaluation pipeline for STL safety monitors.

Runs STL monitors (and baselines) over logged trajectories,
computes robustness statistics, and generates comparison tables.
"""
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.monitors.stl_monitor import STLSafetyMonitor
from src.monitors.baselines import ThresholdMonitor, AnomalyDetector, CBFMonitor
from src.training.trajectory_logger import TrajectoryLogger


def load_config(config_path: str) -> dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def evaluate_stl_monitor(
    monitor: STLSafetyMonitor,
    trajectory: dict[str, list[float]],
) -> dict:
    """Evaluate STL monitor on a single trajectory.

    Returns:
        Dict with per-property robustness and composite score.
    """
    results = monitor.evaluate_all_trajectory(trajectory)
    score = monitor.get_safety_score(results)

    return {
        "stl_results": results,
        "safety_score": score,
        "all_satisfied": all(
            r.get("satisfied", False) for r in results.values()
            if r.get("robustness") is not None
        ),
    }


def evaluate_threshold_monitor(
    monitor: ThresholdMonitor,
    trajectory: dict[str, list[float]],
) -> dict:
    """Evaluate threshold monitor on a single trajectory."""
    return monitor.check_trajectory(trajectory)


def evaluate_anomaly_detector(
    detector: AnomalyDetector,
    trajectory: dict[str, list[float]],
) -> dict:
    """Evaluate anomaly detector on a single trajectory."""
    # Stack trajectory into array
    keys = [k for k in trajectory if k in ["body_roll", "body_pitch", "body_height",
                                             "body_velocity_x", "body_velocity_y"]]
    data = np.column_stack([trajectory[k] for k in keys])
    return detector.check_trajectory(data)


def run_full_evaluation(
    trajectories_dir: str,
    stl_config_path: str,
    output_dir: str = "data/results",
):
    """Run complete evaluation pipeline across all trajectories.

    This function:
    1. Loads all logged trajectories
    2. Runs STL monitor, threshold monitor, anomaly detector, CBF
    3. Computes per-seed statistics
    4. Generates comparison tables

    Args:
        trajectories_dir: Directory containing parquet trajectory files.
        stl_config_path: Path to STL properties YAML config.
        output_dir: Directory to save results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load configs
    stl_config = load_config(stl_config_path)

    # Initialize monitors
    stl_monitor = STLSafetyMonitor(stl_config["properties"])

    threshold_monitor = ThresholdMonitor({
        "body_roll": 0.5,
        "body_pitch": 0.5,
        "body_height": 0.2,
    })

    cbf_monitor = CBFMonitor(
        state_dim=6,
        safe_region={
            "body_roll": (-0.5, 0.5),
            "body_pitch": (-0.5, 0.5),
            "body_height": (0.2, 5.0),
        },
    )

    # Load trajectories
    traj_logger = TrajectoryLogger(trajectories_dir)
    all_trajectories = traj_logger.load_all()

    if not all_trajectories:
        print(f"No trajectories found in {trajectories_dir}")
        return

    print(f"Loaded {len(all_trajectories)} trajectories")

    # Evaluate each trajectory with all monitors
    all_results = []
    for i, df in enumerate(all_trajectories):
        signal_data = traj_logger.to_signal_data(df)

        stl_result = evaluate_stl_monitor(stl_monitor, signal_data)
        threshold_result = evaluate_threshold_monitor(threshold_monitor, signal_data)
        cbf_result = cbf_monitor.check_trajectory(signal_data)

        all_results.append({
            "trajectory_index": i,
            "stl": stl_result,
            "threshold": threshold_result,
            "cbf": cbf_result,
        })

    # Aggregate results
    stl_scores = [r["stl"]["safety_score"] for r in all_results]
    threshold_violations = [r["threshold"]["violation_rate"] for r in all_results]
    cbf_violations = [r["cbf"]["violation_rate"] for r in all_results]

    summary = {
        "n_trajectories": len(all_results),
        "stl": {
            "mean_score": float(np.mean(stl_scores)),
            "std_score": float(np.std(stl_scores)),
            "min_score": float(np.min(stl_scores)),
            "violation_rate": float(np.mean([not r["stl"]["all_satisfied"] for r in all_results])),
        },
        "threshold": {
            "mean_violation_rate": float(np.mean(threshold_violations)),
            "std_violation_rate": float(np.std(threshold_violations)),
        },
        "cbf": {
            "mean_violation_rate": float(np.mean(cbf_violations)),
            "std_violation_rate": float(np.std(cbf_violations)),
        },
    }

    # Save results
    with open(output_path / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    with open(output_path / "evaluation_detailed.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame([
        {
            "Monitor": "STL",
            "Mean Safety Score": f"{summary['stl']['mean_score']:.4f}",
            "Std Safety Score": f"{summary['stl']['std_score']:.4f}",
            "Violation Rate": f"{summary['stl']['violation_rate']:.2%}",
        },
        {
            "Monitor": "Threshold",
            "Mean Safety Score": "N/A",
            "Std Safety Score": "N/A",
            "Violation Rate": f"{summary['threshold']['mean_violation_rate']:.2%}",
        },
        {
            "Monitor": "CBF",
            "Mean Safety Score": "N/A",
            "Std Safety Score": "N/A",
            "Violation Rate": f"{summary['cbf']['mean_violation_rate']:.2%}",
        },
    ])

    comparison_df.to_csv(output_path / "comparison_table.csv", index=False)
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate STL safety monitors")
    parser.add_argument("--trajectories_dir", type=str, default="data/trajectories")
    parser.add_argument("--stl_config", type=str, default="configs/stl_properties.yaml")
    parser.add_argument("--output_dir", type=str, default="data/results")
    args = parser.parse_args()

    run_full_evaluation(args.trajectories_dir, args.stl_config, args.output_dir)


if __name__ == "__main__":
    main()
