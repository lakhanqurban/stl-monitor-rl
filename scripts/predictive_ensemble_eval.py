"""Run predictive STL evaluation with ensemble."""
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
N_EPISODES = 5
MAX_STEPS = 500

envs = {
    "Ant-v5": {"obs_dim": 105, "action_dim": 8, "config": "stl_properties.yaml"},
    "HalfCheetah-v5": {"obs_dim": 17, "action_dim": 6, "config": "stl_properties_halfcheetah.yaml"},
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


def load_ensemble(env_name, obs_dim, action_dim):
    ensemble_path = f"{PROJECT_ROOT}/data/ensemble_{env_name}.pt"
    saved = torch.load(ensemble_path, map_location=device)
    ensemble = []
    for item in saved:
        wm = WorldModel(obs_dim, action_dim, latent_dim=128).to(device)
        wm.load_state_dict(item["model_state_dict"])
        wm.eval()
        ensemble.append(wm)
    return ensemble


def load_policy(env_name):
    env = gym.make(env_name)
    checkpoint_dir = "checkpoints_halfcheetah" if env_name == "HalfCheetah-v5" else "checkpoints"
    for s in [0, 42, 123, 456, 789]:
        p = f"{PROJECT_ROOT}/data/{checkpoint_dir}/seed_{s:03d}/final_model"
        if os.path.exists(p + ".zip"):
            policy = PPO.load(p)
            env.close()
            return policy
    policy = PPO("MlpPolicy", env, verbose=0)
    env.close()
    return policy


def get_stl_signals(obs, env_name):
    """Extract STL signals from observation."""
    if env_name == "Ant-v5":
        return {
            "body_height": float(obs[2]),
            "body_roll": float(obs[4]),
            "body_pitch": float(obs[5]),
            "body_velocity_x": float(obs[0]),
            "n_airborne_feet": 2.0,
        }
    else:
        # HalfCheetah-v5: obs=[x, z, y, torso_angle, hip1, knee1, hip2, knee2,
        #                      ang_vel_x, ang_vel_y, ang_vel_z, ang_vel_hip1, ang_vel_knee1, ang_vel_hip2, ang_vel_knee2,
        #                      vel_x, vel_y, vel_z, vel_hip1, vel_knee1, vel_hip2, vel_knee2]
        return {
            "body_height": float(obs[1]),
            "body_roll": 0.0,
            "body_pitch": float(obs[3]),
            "body_velocity_x": float(obs[8]) if len(obs) > 8 else 0.0,
            "n_airborne_feet": 2.0,
        }


def eval_stl_on_signals(stl_monitor, signals_dict):
    """Evaluate STL on a single-timestep signal dict."""
    scores = {}
    for prop_name, prop in stl_monitor.monitors.items():
        try:
            signal_data = {}
            var_mapping = prop["config"].get("variables", {})
            inverse_mapping = {v: k for k, v in var_mapping.items()}
            for obs_name, value in signals_dict.items():
                stl_var = inverse_mapping.get(obs_name, obs_name)
                signal_data[stl_var] = [value]
            result = stl_monitor.evaluate_trajectory(prop_name, signal_data)
            scores[prop_name] = result["robustness"]
        except Exception:
            scores[prop_name] = 0.0
    return scores


def eval_stl_on_sequence(stl_monitor, obs_seq, env_name):
    """Evaluate STL on a sequence of observations."""
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


def run_experiment(env_name, ensemble, stl_monitor, policy, condition, dims):
    """Run one condition across all horizons."""
    force_mag = condition[1].get("force_mag", 0.0)
    noise_std = condition[1].get("noise_std", 0.0)
    mass_scale = condition[1].get("mass_scale", 1.0)

    metrics = {K: {"reactive": [], "predictive": [], "uncertainty": [], "pred_error": [], "future_viol": []}
               for K in HORIZONS}

    for ep in range(N_EPISODES):
        env = gym.make(env_name)
        if mass_scale != 1.0:
            env.unwrapped.model.body_mass[:] *= mass_scale

        obs, _ = env.reset()
        ep_obs = [obs.copy()]
        ep_actions = []

        for t in range(MAX_STEPS):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if noise_std > 0:
                obs = obs + np.random.normal(0, noise_std, obs.shape).astype(obs.dtype)
            ep_obs.append(obs.copy())
            ep_actions.append(action.copy())
            if terminated or truncated:
                break

        env.close()
        ep_obs = np.array(ep_obs)
        ep_actions = np.array(ep_actions)
        T = len(ep_actions)

        for K in HORIZONS:
            for t in range(min(T - K, MAX_STEPS - K)):
                # Reactive: evaluate on observed window
                window = ep_obs[t:t + K + 1]
                reactive_scores = eval_stl_on_sequence(stl_monitor, window, env_name)
                reactive_mean = np.mean(list(reactive_scores.values()))
                metrics[K]["reactive"].append(reactive_mean)

                # Ensemble predictions
                init_obs = ep_obs[t]
                act_seq = ep_actions[t:t + K]
                if len(act_seq) < K:
                    act_seq = np.pad(act_seq, ((0, K - len(act_seq)), (0, 0)), mode='edge')

                init_t = torch.FloatTensor(init_obs).unsqueeze(0).to(device)
                act_t = torch.FloatTensor(act_seq).unsqueeze(0).to(device)

                member_scores = []
                for wm in ensemble:
                    with torch.no_grad():
                        pred_obs, _ = wm.predict_trajectory(init_t, act_t, K)
                    pred_np = pred_obs.cpu().numpy()[0]
                    pred_scores = eval_stl_on_sequence(stl_monitor, pred_np, env_name)
                    member_scores.append(np.mean(list(pred_scores.values())))

                member_scores = np.array(member_scores)
                pred_mean = member_scores.mean()
                pred_std = member_scores.std()

                metrics[K]["predictive"].append(pred_mean)
                metrics[K]["uncertainty"].append(pred_std)

                # Prediction error
                actual = ep_obs[t + 1:t + K + 1]
                if len(actual) == K:
                    # Get mean prediction
                    preds_all = []
                    for wm in ensemble:
                        with torch.no_grad():
                            p, _ = wm.predict_trajectory(init_t, act_t, K)
                        preds_all.append(p.cpu().numpy()[0])
                    pred_mean_obs = np.mean(preds_all, axis=0)
                    error = np.mean(np.abs(pred_mean_obs - actual))
                    metrics[K]["pred_error"].append(error)

                    # Future violation
                    has_viol = False
                    for tt in range(t + 1, min(t + K + 1, T)):
                        future_window = ep_obs[tt:tt + 2]
                        if len(future_window) >= 2:
                            future_scores = eval_stl_on_sequence(stl_monitor, future_window, env_name)
                            if np.mean(list(future_scores.values())) < 0:
                                has_viol = True
                                break
                    metrics[K]["future_viol"].append(has_viol)

    # Compute summary
    summary = {}
    for K in HORIZONS:
        m = metrics[K]
        n = len(m["reactive"])
        if n == 0:
            summary[K] = {"n": 0}
            continue

        reactive = np.array(m["reactive"])
        predictive = np.array(m["predictive"])
        uncertainty = np.array(m["uncertainty"])
        pred_error = np.array(m["pred_error"]) if m["pred_error"] else np.array([0.0])
        future_viol = np.array(m["future_viol"]) if m["future_viol"] else np.array([False])

        # Correlations
        if len(uncertainty) == len(pred_error) and len(uncertainty) > 2:
            corr_ue = float(np.corrcoef(uncertainty, pred_error)[0, 1])
            if np.isnan(corr_ue):
                corr_ue = 0.0
        else:
            corr_ue = 0.0

        if len(uncertainty) == len(future_viol) and len(uncertainty) > 2:
            corr_uv = float(np.corrcoef(uncertainty, future_viol.astype(float))[0, 1])
            if np.isnan(corr_uv):
                corr_uv = 0.0
        else:
            corr_uv = 0.0

        summary[K] = {
            "n": n,
            "reactive_mean": float(reactive.mean()),
            "reactive_std": float(reactive.std()),
            "predictive_mean": float(predictive.mean()),
            "predictive_std": float(predictive.std()),
            "delta_rho_mean": float(np.abs(predictive - reactive).mean()),
            "uncertainty_mean": float(uncertainty.mean()),
            "pred_error_mean": float(pred_error.mean()),
            "corr_ue": corr_ue,
            "corr_uv": corr_uv,
            "violation_rate": float(future_viol.sum() / len(future_viol)),
        }

    return summary


if __name__ == "__main__":
    all_results = {}

    for env_name, dims in envs.items():
        print(f"\n{'='*60}")
        print(f"Ensemble Predictive Evaluation: {env_name}")
        print(f"{'='*60}")

        ensemble = load_ensemble(env_name, dims["obs_dim"], dims["action_dim"])
        policy = load_policy(env_name)

        with open(f"{PROJECT_ROOT}/configs/{dims['config']}") as f:
            stl_config = yaml.safe_load(f)
        stl_monitor = STLSafetyMonitor(stl_config["properties"])

        env_results = {}
        for cond_name, cond_kwargs in conditions:
            print(f"\n--- {cond_name} ---")
            t0 = time.time()
            summary = run_experiment(
                env_name, ensemble, stl_monitor, policy,
                (cond_name, cond_kwargs), dims
            )
            env_results[cond_name] = summary
            elapsed = time.time() - t0

            for K in HORIZONS:
                s = summary[K]
                if s.get("n", 0) > 0:
                    print(f"  K={K:2d}: R={s['reactive_mean']:+.3f} P={s['predictive_mean']:+.3f} "
                          f"unc={s['uncertainty_mean']:.4f} err={s['pred_error_mean']:.3f} "
                          f"corr_ue={s['corr_ue']:+.3f} corr_uv={s['corr_uv']:+.3f} "
                          f"viol={s['violation_rate']:.1%} [{elapsed:.0f}s]")

        all_results[env_name] = env_results

    out_path = f"{PROJECT_ROOT}/data/results/ensemble_predictive_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")
