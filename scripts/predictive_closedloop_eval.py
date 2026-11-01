"""Closed-loop predictive STL evaluation.

At each prediction step, the predicted observation is fed to the policy
to obtain the next action (closed-loop rollout), rather than using
recorded actions from the environment (open-loop).
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
from src.monitors.stl_monitor import STLSafetyMonitor
from scripts.world_model_train import WorldModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

HORIZONS = [1, 5, 10, 20]
N_EPISODES = 3
MAX_EVAL_TSTEPS = 100

ENV_CONFIGS = {
    "Ant-v5": {"obs_dim": 105, "action_dim": 8, "config": "stl_properties.yaml", "ckpt_dir": "checkpoints"},
    "HalfCheetah-v5": {"obs_dim": 17, "action_dim": 6, "config": "stl_properties_halfcheetah.yaml", "ckpt_dir": "checkpoints_halfcheetah"},
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


def get_stl_signals(obs, env_name):
    if env_name == "Ant-v5":
        return {
            "body_height": float(obs[2]),
            "body_roll": float(obs[4]),
            "body_pitch": float(obs[5]),
            "body_velocity_x": float(obs[0]),
            "n_airborne_feet": 2.0,
        }
    else:
        return {
            "body_height": float(obs[1]),
            "body_roll": 0.0,
            "body_pitch": float(obs[3]),
            "body_velocity_x": float(obs[8]) if len(obs) > 8 else 0.0,
            "n_airborne_feet": 2.0,
        }


def eval_stl_on_sequence(stl_monitor, obs_seq, env_name):
    all_signals = {}
    for obs in obs_seq:
        sig = get_stl_signals(obs, env_name)
        for k, v in sig.items():
            all_signals.setdefault(k, []).append(v)
    scores = {}
    for prop_name, prop in stl_monitor.monitors.items():
        try:
            signal_data = {}
            var_mapping = prop["config"].get("variables", {})
            inverse_mapping = {v: k for k, v in var_mapping.items()}
            for obs_name, values in all_signals.items():
                stl_var = inverse_mapping.get(obs_name, obs_name)
                signal_data[stl_var] = values
            result = stl_monitor.evaluate_trajectory(prop_name, signal_data)
            scores[prop_name] = result["robustness"]
        except Exception:
            scores[prop_name] = 0.0
    return scores


def closed_loop_predict(ensemble, policy, initial_obs, horizon, env_name):
    """Closed-loop prediction: policy acts on predicted observations.
    
    Returns:
        all_predicted_obs: (M, horizon+1, obs_dim) including initial
        all_predicted_actions: (M, horizon, action_dim)
    """
    init_t = torch.FloatTensor(initial_obs).unsqueeze(0).to(device)
    
    all_preds = []
    all_actions = []
    
    for wm in ensemble:
        wm.eval()
        current_obs = init_t
        predicted_obs = [initial_obs.copy()]
        predicted_actions = []
        
        # Get initial action from policy on observed state
        action, _ = policy.predict(initial_obs, deterministic=True)
        last_action_t = torch.FloatTensor(action).unsqueeze(0).to(device)
        
        for k in range(horizon):
            # World model predicts next state given current obs + action
            with torch.no_grad():
                latent = wm.encoder(current_obs)
                next_latent, _ = wm.dynamics(latent, last_action_t)
                next_obs_pred = wm.decoder(next_latent)
            
            next_obs_np = next_obs_pred.cpu().numpy()[0]
            predicted_obs.append(next_obs_np)
            
            # Policy acts on predicted observation (closed-loop)
            action, _ = policy.predict(next_obs_np, deterministic=True)
            predicted_actions.append(action)
            last_action_t = torch.FloatTensor(action).unsqueeze(0).to(device)
            current_obs = next_obs_pred
        
        all_preds.append(np.array(predicted_obs))
        all_actions.append(np.array(predicted_actions))
    
    return np.array(all_preds), np.array(all_actions)


def run_env(env_name, dims):
    print(f"\n{'='*60}")
    print(f"  CLOSED-LOOP: {env_name}")
    print(f"{'='*60}")

    # Load ensemble
    saved = torch.load(f"{PROJECT_ROOT}/data/ensemble_{env_name}.pt", map_location=device)
    ensemble = []
    for item in saved:
        wm = WorldModel(dims["obs_dim"], dims["action_dim"], latent_dim=128).to(device)
        wm.load_state_dict(item["model_state_dict"])
        wm.eval()
        ensemble.append(wm)

    # Load policy
    policy = PPO.load(f"{PROJECT_ROOT}/data/{dims['ckpt_dir']}/seed_000/final_model")

    # Load STL
    with open(f"{PROJECT_ROOT}/configs/{dims['config']}") as f:
        stl_config = yaml.safe_load(f)
    stl_monitor = STLSafetyMonitor(stl_config["properties"])

    all_results = {}
    for cond_name, cond_kwargs in conditions:
        force_mag = cond_kwargs.get("force_mag", 0.0)
        noise_std = cond_kwargs.get("noise_std", 0.0)
        mass_scale = cond_kwargs.get("mass_scale", 1.0)

        print(f"\n--- {cond_name} ---", end="", flush=True)
        t0 = time.time()

        metrics = {K: {"oracle": [], "predictive": [], "uncertainty": [], "pred_error": [], "future_viol": []}
                   for K in HORIZONS}

        for ep in range(N_EPISODES):
            env = gym.make(env_name)
            if mass_scale != 1.0:
                env.unwrapped.model.body_mass[:] *= mass_scale

            obs, _ = env.reset()
            ep_obs = [obs.copy()]

            for t in range(MAX_EVAL_TSTEPS):
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                if noise_std > 0:
                    obs = obs + np.random.normal(0, noise_std, obs.shape).astype(obs.dtype)
                ep_obs.append(obs.copy())
                if terminated or truncated:
                    break

            env.close()
            ep_obs = np.array(ep_obs)
            T = len(ep_obs) - 1  # number of actions

            for K in HORIZONS:
                step = max(1, K)
                for t in range(0, min(T - K, MAX_EVAL_TSTEPS), step):
                    # Oracle: evaluate on actually observed future trajectory
                    oracle_window = ep_obs[t:t + K + 1]
                    oracle_scores = eval_stl_on_sequence(stl_monitor, oracle_window, env_name)
                    oracle_mean = np.mean(list(oracle_scores.values()))
                    metrics[K]["oracle"].append(oracle_mean)

                    # Closed-loop predictive
                    init_obs = ep_obs[t]
                    cl_preds, cl_actions = closed_loop_predict(
                        ensemble, policy, init_obs, K, env_name
                    )
                    
                    # Per-member STL scores on predicted trajectories
                    member_scores = []
                    for m in range(len(ensemble)):
                        pred_scores = eval_stl_on_sequence(stl_monitor, cl_preds[m], env_name)
                        member_scores.append(np.mean(list(pred_scores.values())))

                    member_scores = np.array(member_scores)
                    pred_mean = member_scores.mean()
                    pred_std = member_scores.std()

                    metrics[K]["predictive"].append(float(pred_mean))
                    metrics[K]["uncertainty"].append(float(pred_std))

                    # Prediction error: compare ensemble mean to actual
                    actual = ep_obs[t + 1:t + K + 1]
                    if len(actual) == K:
                        pred_mean_obs = cl_preds.mean(axis=0)[1:]  # skip initial
                        error = np.mean(np.abs(pred_mean_obs - actual))
                        metrics[K]["pred_error"].append(float(error))

                        # Future violation: does oracle score drop below 0?
                        has_viol = False
                        for tt in range(t + 1, min(t + K + 1, T)):
                            future_window = ep_obs[tt:tt + 2]
                            if len(future_window) >= 2:
                                future_scores = eval_stl_on_sequence(stl_monitor, future_window, env_name)
                                if np.mean(list(future_scores.values())) < 0:
                                    has_viol = True
                                    break
                        metrics[K]["future_viol"].append(has_viol)

        elapsed = time.time() - t0

        summary = {}
        for K in HORIZONS:
            m = metrics[K]
            n = len(m["oracle"])
            if n == 0:
                summary[K] = {"n": 0}
                continue

            oracle = np.array(m["oracle"])
            predictive = np.array(m["predictive"])
            uncertainty = np.array(m["uncertainty"])
            pred_error = np.array(m["pred_error"]) if m["pred_error"] else np.array([0.0])
            future_viol = np.array(m["future_viol"]) if m["future_viol"] else np.array([False])

            corr_ue = 0.0
            if len(uncertainty) == len(pred_error) and len(uncertainty) > 2:
                corr_ue = float(np.corrcoef(uncertainty, pred_error)[0, 1])
                if np.isnan(corr_ue): corr_ue = 0.0

            corr_uv = 0.0
            if len(uncertainty) == len(future_viol) and len(uncertainty) > 2:
                corr_uv = float(np.corrcoef(uncertainty, future_viol.astype(float))[0, 1])
                if np.isnan(corr_uv): corr_uv = 0.0

            summary[K] = {
                "n": n,
                "oracle_mean": float(oracle.mean()),
                "oracle_std": float(oracle.std()),
                "predictive_mean": float(predictive.mean()),
                "predictive_std": float(predictive.std()),
                "delta_rho_mean": float(np.abs(predictive - oracle).mean()),
                "uncertainty_mean": float(uncertainty.mean()),
                "pred_error_mean": float(pred_error.mean()),
                "corr_ue": corr_ue,
                "corr_uv": corr_uv,
                "violation_rate": float(future_viol.sum() / len(future_viol)),
            }

        all_results[cond_name] = summary

        for K in HORIZONS:
            s = summary[K]
            if s.get("n", 0) > 0:
                print(f"\n  K={K:2d}: O={s['oracle_mean']:+.3f} P={s['predictive_mean']:+.3f} "
                      f"unc={s['uncertainty_mean']:.4f} err={s['pred_error_mean']:.3f} "
                      f"corr_ue={s['corr_ue']:+.3f} corr_uv={s['corr_uv']:+.3f} "
                      f"viol={s['violation_rate']:.1%}", end="")
        print(f"  [{elapsed:.0f}s]")

    return all_results


if __name__ == "__main__":
    all_env_results = {}
    for env_name, dims in ENV_CONFIGS.items():
        all_env_results[env_name] = run_env(env_name, dims)

    out_path = f"{PROJECT_ROOT}/data/results/closedloop_predictive_results.json"
    with open(out_path, "w") as f:
        json.dump(all_env_results, f, indent=2)
    print(f"\nSaved combined results to {out_path}")
