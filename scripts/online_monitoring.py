"""Phase 7: Online monitoring with per-timestep RTAMT evaluation.

Measures:
- Computational overhead per step
- Intervention latency (time from violation to detection)
- Comparison: offline post-hoc vs. online real-time monitoring
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
from src.monitors.stl_monitor import STLSafetyMonitor
import yaml

ENV_ID = "Ant-v5"
OUTPUT_DIR = "data/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("configs/stl_properties.yaml") as f:
    stl_config = yaml.safe_load(f)


class OnlineSTLMonitor:
    """Online STL monitor that evaluates per timestep and measures overhead."""

    def __init__(self, properties_config: dict):
        self.properties_config = properties_config
        self.monitors = {}
        self._build_monitors()

    def _build_monitors(self):
        for prop_name, prop_config in self.properties_config.items():
            self.monitors[prop_name] = self._create_monitor(prop_config)

    def _create_monitor(self, prop_config: dict):
        spec = __import__("rtamt").StlDiscreteTimeSpecification()
        spec.name = prop_config["name"]
        for var_name in prop_config.get("variables", {}).keys():
            spec.declare_var(var_name, "float")
        formula = prop_config["formula"]
        params = prop_config.get("parameters", {})
        for pname, pval in params.items():
            formula = formula.replace(f"{{{pname}}}", str(pval))
        spec.spec = formula
        spec.parse()
        spec.pastify()
        return {"spec": spec, "config": prop_config}

    def reset(self):
        """Create fresh monitor instances for a new episode."""
        self._build_monitors()

    def step(self, timestep: int, readings: dict[str, float]) -> dict:
        """Evaluate all properties at a single timestep. Returns per-step results."""
        results = {}
        for prop_name, monitor in self.monitors.items():
            var_mapping = monitor["config"].get("variables", {})
            inverse_mapping = {v: k for k, v in var_mapping.items()}
            signal = []
            for obs_name, value in readings.items():
                stl_var = inverse_mapping.get(obs_name, obs_name)
                signal.append((stl_var, float(value)))
            try:
                rob = monitor["spec"].update(timestep, signal)
            except Exception:
                rob = 0.0
            results[prop_name] = {
                "robustness": rob,
                "satisfied": rob >= 0,
            }
        return results


def run_online_monitoring(model, n_episodes=5, max_steps=500):
    """Run online monitoring and measure overhead + latency."""
    online = OnlineSTLMonitor(stl_config["properties"])
    offline = STLSafetyMonitor(stl_config["properties"])

    overhead_per_step = []
    intervention_latencies = []

    for ep in range(n_episodes):
        raw_env = gym.make(ENV_ID, render_mode=None)
        wrapped = LocomotionWrapper(raw_env)
        obs, info = wrapped.reset(seed=ep * 100)
        online.reset()

        step_times = []
        robustness_traces = {p: [] for p in online.monitors}
        first_violation_step = None
        detected_step = None

        for t in range(max_steps):
            t_start = time.perf_counter()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped.step(action)

            mon_obs = info.get("monitor_obs", {})
            readings = {k: float(v[0]) if hasattr(v, '__len__') else float(v)
                        for k, v in mon_obs.items()
                        if k in ["body_roll", "body_pitch", "body_height",
                                 "body_velocity_x", "body_velocity_y"]}

            if readings:
                step_results = online.step(t, readings)
                for prop_name, res in step_results.items():
                    robustness_traces[prop_name].append(res["robustness"])

                    if first_violation_step is None and not res["satisfied"]:
                        first_violation_step = t
                    if detected_step is None and not res["satisfied"]:
                        detected_step = t

            t_end = time.perf_counter()
            step_times.append(t_end - t_start)

            if terminated or truncated:
                break

        overhead_per_step.append(step_times)

        if first_violation_step is not None and detected_step is not None:
            intervention_latencies.append(detected_step - first_violation_step)
        elif first_violation_step is None:
            intervention_latencies.append(0)

        # Offline evaluation for comparison
        if online.monitors:
            signal_data = {}
            for t_idx in range(min(len(robustness_traces[list(robustness_traces.keys())[0]]), max_steps)):
                mon_obs = info.get("monitor_obs", {})
                readings = {k: float(v[0]) if hasattr(v, '__len__') else float(v)
                            for k, v in mon_obs.items()
                            if k in ["body_roll", "body_pitch", "body_height",
                                     "body_velocity_x", "body_velocity_y"]}
                for k, v in readings.items():
                    if k not in signal_data:
                        signal_data[k] = []
                    signal_data[k].append(v)

        raw_env.close()

    return {
        "overhead_per_step_ms": [np.mean(s) * 1000 for s in overhead_per_step],
        "overhead_std_ms": [np.std(s) * 1000 for s in overhead_per_step],
        "mean_overhead_ms": float(np.mean([np.mean(s) for s in overhead_per_step]) * 1000),
        "std_overhead_ms": float(np.std([np.mean(s) for s in overhead_per_step]) * 1000),
        "intervention_latencies": intervention_latencies,
        "mean_latency_steps": float(np.mean(intervention_latencies)),
        "n_episodes": n_episodes,
        "n_violations_detected": sum(1 for l in intervention_latencies if l > 0),
    }


def run_offline_benchmark(model, n_episodes=5, max_steps=500):
    """Benchmark offline post-hoc evaluation."""
    offline = STLSafetyMonitor(stl_config["properties"])

    episode_times = []
    for ep in range(n_episodes):
        raw_env = gym.make(ENV_ID, render_mode=None)
        wrapped = LocomotionWrapper(raw_env)
        obs, info = wrapped.reset(seed=ep * 100)

        signal_data = {}
        for t in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            mon_obs = info.get("monitor_obs", {})
            readings = {k: float(v[0]) if hasattr(v, '__len__') else float(v)
                        for k, v in mon_obs.items()
                        if k in ["body_roll", "body_pitch", "body_height",
                                 "body_velocity_x", "body_velocity_y"]}
            for k, v in readings.items():
                if k not in signal_data:
                    signal_data[k] = []
                signal_data[k].append(v)
            if terminated or truncated:
                break
        raw_env.close()

        t_start = time.perf_counter()
        results = offline.evaluate_all_trajectory(signal_data)
        t_end = time.perf_counter()
        episode_times.append(t_end - t_start)

    return {
        "mean_episode_time_ms": float(np.mean(episode_times) * 1000),
        "std_episode_time_ms": float(np.std(episode_times) * 1000),
        "n_episodes": n_episodes,
    }


if __name__ == "__main__":
    model_path = None
    for s in [42, 0, 123, 456, 789]:
        p = f"data/checkpoints/seed_{s:03d}/final_model.zip"
        if os.path.exists(p):
            model_path = p
            break
    if not model_path:
        print("No trained model found.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    print("\n--- Online Monitoring ---")
    online_results = run_online_monitoring(model, n_episodes=5, max_steps=500)
    print(f"  Mean overhead: {online_results['mean_overhead_ms']:.3f} ms/step")
    print(f"  Mean intervention latency: {online_results['mean_latency_steps']:.1f} steps")
    print(f"  Violations detected: {online_results['n_violations_detected']}/{online_results['n_episodes']}")

    print("\n--- Offline Post-Hoc ---")
    offline_results = run_offline_benchmark(model, n_episodes=5, max_steps=500)
    print(f"  Mean episode eval time: {offline_results['mean_episode_time_ms']:.3f} ms")

    comparison = {
        "online": online_results,
        "offline": offline_results,
        "speedup_factor": offline_results["mean_episode_time_ms"] / (online_results["mean_overhead_ms"] * 500 + 1e-9),
    }

    with open(f"{OUTPUT_DIR}/online_monitoring.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"ONLINE vs OFFLINE COMPARISON")
    print(f"{'='*60}")
    print(f"  Online per-step overhead:  {online_results['mean_overhead_ms']:.3f} ms")
    print(f"  Offline per-episode time:  {offline_results['mean_episode_time_ms']:.3f} ms")
    print(f"  Intervention latency:      {online_results['mean_latency_steps']:.1f} steps")
    print(f"  Offline/Online speedup:    {comparison['speedup_factor']:.1f}x")
