"""Train ensemble of world models and run predictive STL experiments.

Trains M=3 world models per environment with different seeds,
then evaluates predictive STL across horizons K={1,5,10,20}
and computes uncertainty metrics.
"""
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from src.environments.locomotion_wrapper import LocomotionWrapper
from src.monitors.stl_monitor import STLSafetyMonitor
from scripts.world_model_train import WorldModel, WorldModelTrainer

ENSEMBLE_SIZE = 3
HORIZONS = [1, 5, 10, 20]
N_EVAL_EPISODES = 5
MAX_STEPS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def train_ensemble(env_name, obs_dim, action_dim, n_models=ENSEMBLE_SIZE, epochs=30):
    """Train ensemble of world models with different random seeds."""
    ensemble = []
    for i in range(n_models):
        print(f"\n--- Training ensemble member {i+1}/{n_models} ---")
        torch.manual_seed(i * 42 + 7)
        np.random.seed(i * 42 + 7)
        
        wm = WorldModel(obs_dim, action_dim, latent_dim=128).to(device)
        trainer = WorldModelTrainer(wm, lr=1e-3)
        
        obs, acts, rews = trainer.collect_data(env_name, n_episodes=50)
        losses = trainer.train(obs, acts, rews, epochs=epochs)
        
        print(f"  Final loss: {losses[-1]:.4f}")
        ensemble.append(wm)
    
    return ensemble


def collect_nominal_data(env_name, n_episodes=50):
    """Collect nominal trajectory data for evaluation."""
    env = gym.make(env_name)
    # Try multiple seeds
    model = None
    for s in [0, 42, 123, 456, 789]:
        p = f"{PROJECT_ROOT}/data/checkpoints/seed_{s:03d}/final_model.zip"
        if os.path.exists(p):
            model = PPO.load(p.replace(".zip", ""))
            break
    if model is None:
        model = PPO("MlpPolicy", env, verbose=0)
    
    all_obs, all_actions, all_next_obs = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        for t in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, _, terminated, truncated, _ = env.step(action)
            all_obs.append(obs.copy())
            all_actions.append(action.copy())
            all_next_obs.append(next_obs.copy())
            obs = next_obs
            if terminated or truncated:
                break
    env.close()
    return np.array(all_obs), np.array(all_actions), np.array(all_next_obs)


def predict_with_ensemble(ensemble, initial_obs, action_seq, horizon):
    """Get predictions from all ensemble members.
    
    Returns:
        all_predicted_obs: (M, horizon, obs_dim) predictions
    """
    M = len(ensemble)
    initial_obs_t = torch.FloatTensor(initial_obs).unsqueeze(0).to(device)
    action_seq_t = torch.FloatTensor(action_seq).unsqueeze(0).to(device)
    
    all_preds = []
    for wm in ensemble:
        wm.eval()
        with torch.no_grad():
            pred_obs, _ = wm.predict_trajectory(initial_obs_t, action_seq_t, horizon)
        all_preds.append(pred_obs.cpu().numpy()[0])  # (horizon, obs_dim)
    
    return np.array(all_preds)  # (M, horizon, obs_dim)


def extract_stl_signals(obs_seq, env_name):
    """Extract STL monitoring signals from observation sequence.
    
    obs_seq: (T, obs_dim) or (M, T, obs_dim)
    """
    if obs_seq.ndim == 3:
        # Take mean across ensemble for signal extraction
        obs_seq = obs_seq.mean(axis=0)
    
    signals = {}
    if env_name == "Ant-v5":
        # Ant-v5 observation layout: qpos[2]=height, qpos[3]=qw, qpos[4]=qx, qpos[5]=qy, qpos[6]=qz
        # qvel[0]=vx, contact forces in contact signals
        signals["body_height"] = obs_seq[:, 2].tolist() if obs_seq.ndim == 2 else [float(obs_seq[2])]
        signals["body_roll"] = obs_seq[:, 4].tolist() if obs_seq.ndim == 2 else [float(obs_seq[4])]
        signals["body_pitch"] = obs_seq[:, 5].tolist() if obs_seq.ndim == 2 else [float(obs_seq[5])]
        signals["body_velocity_x"] = obs_seq[:, 0].tolist() if obs_seq.ndim == 2 else [float(obs_seq[0])]
        signals["n_airborne_feet"] = [2.0] * (obs_seq.shape[0] if obs_seq.ndim == 2 else 1)
    else:  # HalfCheetah
        signals["body_height"] = obs_seq[:, 0].tolist() if obs_seq.ndim == 2 else [float(obs_seq[0])]
        signals["body_roll"] = [0.0] * (obs_seq.shape[0] if obs_seq.ndim == 2 else 1)
        signals["body_pitch"] = obs_seq[:, 1].tolist() if obs_seq.ndim == 2 else [float(obs_seq[1])]
        signals["body_velocity_x"] = obs_seq[:, 8].tolist() if obs_seq.ndim == 2 and obs_seq.shape[1] > 8 else [0.0]
        signals["n_airborne_feet"] = [2.0] * (obs_seq.shape[0] if obs_seq.ndim == 2 else 1)
    
    return signals


def evaluate_stl_on_signals(stl_monitor, signals):
    """Evaluate STL properties on monitoring signals."""
    results = {}
    for prop_name in stl_monitor.monitors:
        try:
            # Create signal_data for evaluate_trajectory
            signal_data = {}
            var_mapping = stl_monitor.monitors[prop_name]["config"].get("variables", {})
            inverse_mapping = {v: k for k, v in var_mapping.items()}
            for obs_name, values in signals.items():
                stl_var_name = inverse_mapping.get(obs_name, obs_name)
                signal_data[stl_var_name] = values
            
            result = stl_monitor.evaluate_trajectory(prop_name, signal_data)
            results[prop_name] = result["robustness"]
        except Exception:
            results[prop_name] = 0.0
    
    return results


def run_predictive_experiment(ensemble, stl_monitor, env_name, condition_name,
                               force_mag=0.0, noise_std=0.0, mass_scale=1.0,
                               horizons=HORIZONS, n_episodes=N_EVAL_EPISODES):
    """Run predictive STL experiment for one condition."""
    # Create perturbed environment
    env = gym.make(env_name)
    if mass_scale != 1.0:
        model_mj = env.unwrapped.model
        for i in range(model_mj.nbody):
            model_mj.body_mass[i] *= mass_scale
    
    # Load policy
    policy = None
    for s in [0, 42, 123, 456, 789]:
        p = f"{PROJECT_ROOT}/data/checkpoints/seed_{s:03d}/final_model.zip"
        if os.path.exists(p):
            policy = PPO.load(p.replace(".zip", ""))
            break
    if policy is None:
        policy = PPO("MlpPolicy", env, verbose=0)
    
    # Collect trajectories
    all_results = {K: {"reactive_scores": [], "predictive_scores": [], "uncertainties": [],
                        "prediction_errors": [], "future_violations": []} for K in horizons}
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        # Apply noise to observations for policy
        if noise_std > 0:
            obs = obs + np.random.normal(0, noise_std, obs.shape).astype(obs.dtype)
        
        # Collect full episode trajectory
        episode_obs = [obs.copy()]
        episode_actions = []
        
        for t in range(MAX_STEPS):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if noise_std > 0:
                obs_noisy = obs + np.random.normal(0, noise_std, obs.shape).astype(obs.dtype)
            else:
                obs_noisy = obs
            episode_obs.append(obs_noisy.copy())
            episode_actions.append(action.copy())
            obs = obs_noisy
            if terminated or truncated:
                break
        
        episode_obs = np.array(episode_obs)
        episode_actions = np.array(episode_actions)
        T = len(episode_actions)
        
        # Evaluate at each timestep with each horizon
        for K in horizons:
            reactive_scores_K = []
            predictive_scores_K = []
            uncertainties_K = []
            prediction_errors_K = []
            future_violations_K = []
            
            for t in range(min(T - K, MAX_STEPS - K)):
                # Reactive STL on observed trajectory
                obs_window = episode_obs[t:t+K+1]
                signals = extract_stl_signals(obs_window, env_name)
                reactive_result = evaluate_stl_on_signals(stl_monitor, signals)
                reactive_score = np.mean([v for v in reactive_result.values()])
                reactive_scores_K.append(reactive_score)
                
                # Predictive STL via ensemble
                initial_obs = episode_obs[t]
                action_seq = episode_actions[t:t+K]
                if len(action_seq) < K:
                    # Pad with last action
                    action_seq = np.pad(action_seq, ((0, K - len(action_seq)), (0, 0)), mode='edge')
                
                ensemble_preds = predict_with_ensemble(ensemble, initial_obs, action_seq, K)
                
                # Per-member STL scores
                member_scores = []
                for m in range(len(ensemble)):
                    pred_signals = extract_stl_signals(ensemble_preds[m], env_name)
                    pred_result = evaluate_stl_on_signals(stl_monitor, pred_signals)
                    member_scores.append(np.mean([v for v in pred_result.values()]))
                
                member_scores = np.array(member_scores)
                predictive_mean = member_scores.mean()
                predictive_std = member_scores.std()
                
                predictive_scores_K.append(predictive_mean)
                uncertainties_K.append(predictive_std)
                
                # Prediction error: compare ensemble mean to actual
                actual_obs = episode_obs[t+1:t+K+1]
                if len(actual_obs) == K:
                    pred_mean = ensemble_preds.mean(axis=0)
                    error = np.mean(np.abs(pred_mean - actual_obs))
                    prediction_errors_K.append(error)
                    
                    # Future violation: does reactive score drop below 0 in the window?
                    future_reactive = []
                    for tt in range(t+1, min(t+K+1, T)):
                        future_obs = episode_obs[tt:tt+2]
                        future_signals = extract_stl_signals(future_obs, env_name)
                        future_result = evaluate_stl_on_signals(stl_monitor, future_signals)
                        future_reactive.append(np.mean([v for v in future_result.values()]))
                    
                    has_future_violation = any(s < 0 for s in future_reactive) if future_reactive else False
                    future_violations_K.append(has_future_violation)
            
            all_results[K]["reactive_scores"].append(reactive_scores_K)
            all_results[K]["predictive_scores"].append(predictive_scores_K)
            all_results[K]["uncertainties"].append(uncertainties_K)
            all_results[K]["prediction_errors"].append(prediction_errors_K)
            all_results[K]["future_violations"].append(future_violations_K)
    
    env.close()
    return all_results


def compute_metrics(results, horizons):
    """Compute summary metrics from experimental results."""
    metrics = {}
    for K in horizons:
        r = results[K]
        
        # Flatten across episodes
        all_reactive = [s for ep in r["reactive_scores"] for s in ep]
        all_predictive = [s for ep in r["predictive_scores"] for s in ep]
        all_uncertainty = [u for ep in r["uncertainties"] for u in ep]
        all_pred_error = [e for ep in r["prediction_errors"] for e in ep]
        all_future_viol = [v for ep in r["future_violations"] for v in ep]
        
        if not all_reactive:
            metrics[K] = {"n_samples": 0}
            continue
        
        reactive_arr = np.array(all_reactive)
        predictive_arr = np.array(all_predictive)
        uncertainty_arr = np.array(all_uncertainty)
        pred_error_arr = np.array(all_pred_error) if all_pred_error else np.array([0.0])
        future_viol_arr = np.array(all_future_viol) if all_future_viol else np.array([False])
        
        # Delta_rho: predictive vs reactive
        min_len = min(len(reactive_arr), len(predictive_arr))
        delta_rho = np.abs(predictive_arr[:min_len] - reactive_arr[:min_len])
        
        # Uncertainty-error correlation
        if len(uncertainty_arr) == len(pred_error_arr) and len(uncertainty_arr) > 2:
            corr_ue = np.corrcoef(uncertainty_arr, pred_error_arr)[0, 1]
        else:
            corr_ue = 0.0
        
        # Uncertainty-future violation correlation
        if len(uncertainty_arr) == len(future_viol_arr) and len(uncertainty_arr) > 2:
            corr_uv = np.corrcoef(uncertainty_arr, future_viol_arr.astype(float))[0, 1]
        else:
            corr_uv = 0.0
        
        # Early warning: when uncertainty is high, is there a future violation?
        n_violations = future_viol_arr.sum()
        n_total = len(future_viol_arr)
        violation_rate = n_violations / n_total if n_total > 0 else 0.0
        
        metrics[K] = {
            "n_samples": len(all_reactive),
            "reactive_mean": float(reactive_arr.mean()),
            "reactive_std": float(reactive_arr.std()),
            "predictive_mean": float(predictive_arr.mean()),
            "predictive_std": float(predictive_arr.std()),
            "delta_rho_mean": float(delta_rho.mean()),
            "delta_rho_std": float(delta_rho.std()),
            "uncertainty_mean": float(uncertainty_arr.mean()),
            "uncertainty_std": float(uncertainty_arr.std()),
            "pred_error_mean": float(pred_error_arr.mean()),
            "pred_error_std": float(pred_error_arr.std()),
            "corr_uncertainty_error": float(corr_ue) if not np.isnan(corr_ue) else 0.0,
            "corr_uncertainty_violation": float(corr_uv) if not np.isnan(corr_uv) else 0.0,
            "future_violation_rate": float(violation_rate),
        }
    
    return metrics


if __name__ == "__main__":
    environments = {
        "Ant-v5": {"obs_dim": 105, "action_dim": 8},
        "HalfCheetah-v5": {"obs_dim": 17, "action_dim": 6},
    }
    
    conditions = [
        ("baseline", {}),
        ("force_5N", {"force_mag": 5.0}),
        ("force_15N", {"force_mag": 15.0}),
        ("noise_0.05", {"noise_std": 0.05}),
        ("noise_0.15", {"noise_std": 0.15}),
        ("mass_0.8x", {"mass_scale": 0.8}),
        ("mass_1.6x", {"mass_scale": 1.6}),
    ]
    
    all_experiment_results = {}
    
    for env_name, dims in environments.items():
        print(f"\n{'='*60}")
        print(f"Ensemble Training: {env_name}")
        print(f"{'='*60}")
        
        ensemble = train_ensemble(env_name, dims["obs_dim"], dims["action_dim"],
                                   n_models=ENSEMBLE_SIZE, epochs=30)
        
        # Load STL config
        config_file = f"{PROJECT_ROOT}/configs/" + (
            "stl_properties_halfcheetah.yaml" if env_name == "HalfCheetah-v5" else "stl_properties.yaml"
        )
        with open(config_file) as f:
            stl_config = yaml.safe_load(f)
        stl_monitor = STLSafetyMonitor(stl_config["properties"])
        
        env_results = {}
        for cond_name, cond_kwargs in conditions:
            print(f"\n--- {env_name} / {cond_name} ---")
            t0 = time.time()
            results = run_predictive_experiment(
                ensemble, stl_monitor, env_name, cond_name, **cond_kwargs
            )
            metrics = compute_metrics(results, HORIZONS)
            env_results[cond_name] = metrics
            elapsed = time.time() - t0
            
            # Print summary
            for K in HORIZONS:
                m = metrics[K]
                if m.get("n_samples", 0) > 0:
                    print(f"  K={K:2d}: reactive={m['reactive_mean']:+.3f} pred={m['predictive_mean']:+.3f} "
                          f"delta={m['delta_rho_mean']:.4f} unc={m['uncertainty_mean']:.4f} "
                          f"corr_ue={m['corr_uncertainty_error']:.3f} viol_rate={m['future_violation_rate']:.1%} "
                          f"[{elapsed:.1f}s]")
        
        all_experiment_results[env_name] = env_results
    
    # Save results
    out_path = f"{PROJECT_ROOT}/data/results/ensemble_predictive_results.json"
    with open(out_path, "w") as f:
        json.dump(all_experiment_results, f, indent=2)
    print(f"\nSaved to {out_path}")
