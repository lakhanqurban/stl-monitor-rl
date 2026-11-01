"""CoRL extension evaluation: U_o/U_z/U_rho baselines, per-spec, AUROC/AUPRC,
calibration, risk-coverage, S_UP, multi-seed, compound shifts, noise ablation, M-ablation.

Closed-loop rollouts. Saves per-window arrays + summary JSON for paper tables.
Usage:
  python scripts/corl_extension_eval.py --env Ant-v5 --seeds 0 --episodes 3 --horizons 10 20 --quick
  python scripts/corl_extension_eval.py --env all --seeds 0 42 123 --episodes 10
"""
import os, sys, json, time, argparse, itertools
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

ENV_CONFIGS = {
    "Ant-v5": {"obs_dim": 105, "action_dim": 8, "config": "stl_properties.yaml", "ckpt_dir": "checkpoints"},
    "HalfCheetah-v5": {"obs_dim": 17, "action_dim": 6, "config": "stl_properties_halfcheetah.yaml", "ckpt_dir": "checkpoints_halfcheetah"},
}

# Base + extension conditions: (name, kwargs)
BASE_CONDS = [
    ("baseline", {}),
    ("force_5N", {"force_mag": 5.0}),
    ("force_15N", {"force_mag": 15.0}),
    ("noise_0.05", {"noise_std": 0.05}),
    ("noise_0.15", {"noise_std": 0.15}),
    ("mass_0.8x", {"mass_scale": 0.8}),
    ("mass_1.6x", {"mass_scale": 1.6}),
]
EXT_CONDS = [
    ("mass1.3_noise0.05", {"mass_scale": 1.3, "noise_std": 0.05}),
    ("force15_noise0.05", {"force_mag": 15.0, "noise_std": 0.05}),
    ("friction_0.5x", {"friction_scale": 0.5}),
]


def get_stl_signals(obs, env_name):
    if env_name == "Ant-v5":
        return {"body_height": float(obs[2]), "body_roll": float(obs[4]),
                "body_pitch": float(obs[5]), "body_velocity_x": float(obs[0]),
                "n_airborne_feet": 2.0}
    return {"body_height": float(obs[1]), "body_roll": 0.0,
            "body_pitch": float(obs[3]),
            "body_velocity_x": float(obs[8]) if len(obs) > 8 else 0.0,
            "n_airborne_feet": 2.0}


def eval_stl_perspec(stl_monitor, obs_seq, env_name):
    all_signals = {}
    for obs in obs_seq:
        sig = get_stl_signals(obs, env_name)
        for k, v in sig.items():
            all_signals.setdefault(k, []).append(v)
    scores = {}
    for prop_name, prop in stl_monitor.monitors.items():
        try:
            var_mapping = prop["config"].get("variables", {})
            inv = {v: k for k, v in var_mapping.items()}
            sd = {}
            for on, vals in all_signals.items():
                sd[inv.get(on, on)] = vals
            r = stl_monitor.evaluate_trajectory(prop_name, sd)
            scores[prop_name] = float(r["robustness"])
        except Exception:
            scores[prop_name] = 0.0
    return scores


def closed_loop_predict(ensemble, policy, init_obs, K, rollout_noise=0.0):
    """Returns preds (M,K+1,D), latents (M,K+1,L), actions."""
    init_t = torch.FloatTensor(init_obs).unsqueeze(0).to(device)
    preds, lats = [], []
    for wm in ensemble:
        wm.eval()
        cur = init_t
        po = [init_obs.copy()]
        la = [wm.encoder(cur).detach().cpu().numpy()[0]]
        act, _ = policy.predict(init_obs, deterministic=True)
        last_a = torch.FloatTensor(act).unsqueeze(0).to(device)
        for _ in range(K):
            with torch.no_grad():
                lat = wm.encoder(cur)
                nl, _ = wm.dynamics(lat, last_a)
                no = wm.decoder(nl)
            non = no.cpu().numpy()[0]
            no_in = non + (np.random.normal(0, rollout_noise, non.shape) if rollout_noise > 0 else 0)
            po.append(non.copy())
            la.append(nl.cpu().numpy()[0].copy())
            act, _ = policy.predict(no_in, deterministic=True)
            last_a = torch.FloatTensor(act).unsqueeze(0).to(device)
            cur = torch.FloatTensor(non).unsqueeze(0).to(device)
        preds.append(np.array(po)); lats.append(np.array(la))
    return np.array(preds), np.array(lats)


def auroc_auprc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan"), float("nan")
    order = np.argsort(-scores)
    lab = labels[order]
    tp = np.cumsum(lab); fp = np.cumsum(1 - lab)
    P = lab.sum(); N = len(lab) - P
    tpr = np.concatenate([[0], tp / P]); fpr = np.concatenate([[0], fp / N])
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auroc = float(trapz(tpr, fpr))
    # AUPRC via step average
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / P
    # fix orientation: trapz(y,x)
    auprc = float(trapz(np.concatenate([[1], prec]), np.concatenate([[0], rec])))
    return auroc, auprc


def risk_coverage(unc, err, viol, coverages=(1.0, 0.9, 0.8, 0.7)):
    unc = np.asarray(unc, float); order = np.argsort(unc)
    out = {}
    n = len(order)
    for c in coverages:
        k = max(1, int(round(n * c)))
        idx = order[:k]
        out[str(c)] = {"n": k,
            "mean_abs_stl_err": float(np.asarray(err, float)[idx].mean()) if len(idx) else float("nan"),
            "viol_rate": float(np.asarray(viol, int)[idx].mean()) if len(idx) else float("nan")}
    return out


def calib_buckets(unc, stl_err, nb=5):
    unc = np.asarray(unc, float); e = np.asarray(stl_err, float)
    if len(unc) < nb:
        return []
    qs = np.quantile(unc, np.linspace(0, 1, nb + 1))
    rows = []
    for i in range(nb):
        m = (unc >= qs[i]) & (unc <= qs[i + 1] if i == nb - 1 else unc < qs[i + 1])
        if m.sum():
            rows.append({"bin": i, "lo": float(qs[i]), "hi": float(qs[i+1]),
                         "n": int(m.sum()), "mean_err": float(e[m].mean())})
    return rows


def run_one(env_name, dims, seed, K, cond_name, ck, n_ep, max_steps, rollout_noise, m_sub):
    import os as _os
    cand = [f"{PROJECT_ROOT}/data/ensemble_{env_name}_M10.pt",
            f"{PROJECT_ROOT}/data/ensemble_{env_name}_M5.pt",
            f"{PROJECT_ROOT}/data/ensemble_{env_name}.pt"]
    src = next(p for p in cand if _os.path.exists(p))
    saved = torch.load(src, map_location=device)
    ens = []
    for item in saved[:m_sub]:
        wm = WorldModel(dims["obs_dim"], dims["action_dim"], latent_dim=128).to(device)
        wm.load_state_dict(item["model_state_dict"]); wm.eval(); ens.append(wm)
    pol = PPO.load(f"{PROJECT_ROOT}/data/{dims['ckpt_dir']}/seed_{seed:03d}/final_model")
    with open(f"{PROJECT_ROOT}/configs/{dims['config']}") as f:
        stl = STLSafetyMonitor(yaml.safe_load(f)["properties"])
    fm = ck.get("force_mag", 0.0); ns = ck.get("noise_std", 0.0)
    ms = ck.get("mass_scale", 1.0); fs = ck.get("friction_scale", 1.0)
    W = []  # per-window records
    for ep in range(n_ep):
        env = gym.make(env_name)
        try:
            if ms != 1.0: env.unwrapped.model.body_mass[:] *= ms
            if fs != 1.0:
                try: env.unwrapped.model.geom_friction[:] *= fs
                except Exception: pass
        except Exception: pass
        obs, _ = env.reset(seed=1000 + ep)
        ep_obs = [obs.copy()]
        for _ in range(max_steps):
            a, _ = pol.predict(obs, deterministic=True)
            if fm != 0.0:
                try:
                    env.unwrapped.data.xfrc_applied[1, 0] = float(fm)
                except Exception:
                    pass
            obs, _, term, trunc, _ = env.step(a)
            try:
                env.unwrapped.data.xfrc_applied[:] = 0
            except Exception:
                pass
            if ns > 0: obs = obs + np.random.normal(0, ns, obs.shape).astype(obs.dtype)
            ep_obs.append(obs.copy())
            if term or trunc: break
        env.close()
        ep_obs = np.array(ep_obs); T = len(ep_obs) - 1
        for t in range(0, min(T - K, max_steps), max(1, K)):
            o_scores = eval_stl_perspec(stl, ep_obs[t:t+K+1], env_name)
            o_comp = float(np.mean(list(o_scores.values())))
            preds, lats = closed_loop_predict(ens, pol, ep_obs[t], K, rollout_noise)
            m_scores = [eval_stl_perspec(stl, preds[m], env_name) for m in range(len(ens))]
            keys = list(o_scores.keys())
            p_mat = np.array([[s[k] for k in keys] for s in m_scores])
            p_bar = p_mat.mean(0); u_rho = float(p_mat.mean(0).mean() * 0 + p_mat[:, :].mean(axis=0).std() if False else 0)
            # composite disagreement = std of composite across members
            comp_m = p_mat.mean(1); u_rho = float(comp_m.std())
            p_comp = float(comp_m.mean())
            # U_o / U_z: mean over K of std across members
            pm = preds[:, 1:, :].mean(0)  # (K,D) mean
            u_o = float(np.mean([preds[:, k, :].std(0).mean() for k in range(1, K+1)]))
            u_z = float(np.mean([lats[:, k, :].std(0).mean() for k in range(1, K+1)]))
            actual = ep_obs[t+1:t+K+1]
            e_l2 = float(np.mean(np.linalg.norm(pm - actual, axis=1))) if len(actual) == K else float("nan")
            # safety-var error (height+pitch)
            try:
                sh = np.mean([abs(get_stl_signals(pm[k], env_name)["body_height"] - get_stl_signals(actual[k], env_name)["body_height"]) for k in range(K)])
                sp = np.mean([abs(get_stl_signals(pm[k], env_name)["body_pitch"] - get_stl_signals(actual[k], env_name)["body_pitch"]) for k in range(K)])
            except Exception:
                sh, sp = float("nan"), float("nan")
            stl_err = abs(p_comp - o_comp)
            y_comp = int(o_comp < 0)
            y_ps = {k: int(v < 0) for k, v in o_scores.items()}
            p_ps = {k: float(v) for k, v in zip(keys, p_bar)}
            W.append({"O": o_comp, "P": p_comp, "U_rho": u_rho, "U_o": u_o, "U_z": u_z,
                      "E_l2": e_l2, "E_h": sh, "E_p": sp, "STL_err": stl_err, "Y": y_comp,
                      "O_ps": o_scores, "P_ps": p_ps, "Y_ps": y_ps})
    return W


def summarize(W):
    if not W: return {"n": 0}
    O = np.array([w["O"] for w in W]); P = np.array([w["P"] for w in W])
    Ur = np.array([w["U_rho"] for w in W]); Uo = np.array([w["U_o"] for w in W])
    Uz = np.array([w["U_z"] for w in W]); E = np.array([w["E_l2"] for w in W])
    SE = np.array([w["STL_err"] for w in W]); Y = np.array([w["Y"] for w in W], int)
    def corr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0: return float("nan")
        return float(np.corrcoef(a[m], b[m])[0, 1])
    s = {"n": len(W), "O_mean": float(O.mean()), "P_mean": float(P.mean()),
         "bias": float((P - O).mean()), "d_rho": float(np.abs(P - O).mean()),
         "U_rho": float(Ur.mean()), "U_o": float(Uo.mean()), "U_z": float(Uz.mean()),
         "E_l2": float(np.nanmean(E)), "STL_err": float(SE.mean()),
         "viol": float(Y.mean()),
         "r_Urho_E": corr(Ur, E), "r_Uo_E": corr(Uo, E), "r_Uz_E": corr(Uz, E),
         "r_Urho_SE": corr(Ur, SE), "r_Urho_Y": corr(Ur, Y)}
    for nm, sc in [("U_rho", Ur), ("U_o", Uo), ("U_z", Uz), ("negP", -P), ("P", P)]:
        ar, ap = auroc_auprc(sc, Y)
        s[f"AUROC_{nm}"] = ar; s[f"AUPRC_{nm}"] = ap
    # Any-violation target: primary failure label (1 if ANY spec violated)
    Ya = np.array([int(any(w["Y_ps"].values())) for w in W], int)
    s["viol_any"] = float(Ya.mean())
    for nm, sc in [("U_rho", Ur), ("U_o", Uo), ("U_z", Uz), ("negP", -P)]:
        ar, ap = auroc_auprc(sc, Ya)
        s[f"ANY_AUROC_{nm}"] = ar; s[f"ANY_AUPRC_{nm}"] = ap
    # S_UP sweep: AUROC of failure via -S_UP (= P-lamU inverted) and viol rate of flagged S_UP<0
    for lam in [0, 1, 2, 5]:
        SUP = P - lam * Ur
        ar, ap = auroc_auprc(-SUP, Y)
        s[f"AUROC_SUP{lam}"] = ar; s[f"AUPRC_SUP{lam}"] = ap
        s[f"SUP{lam}_mean"] = float(SUP.mean())
        s[f"SUP{lam}_flagrate"] = float((SUP < 0).mean())
    s["risk_coverage_Urho"] = risk_coverage(Ur, SE, Y)
    s["calib_Urho"] = calib_buckets(Ur, SE)
    # per-spec violation rates + AUROC per spec from its own disagreement proxy (composite U as fallback)
    keys = list(W[0]["O_ps"].keys())
    ps = {}
    for k in keys:
        yk = np.array([w["Y_ps"][k] for w in W], int)
        pk = np.array([w["P_ps"][k] for w in W])
        ar, ap = auroc_auprc(-pk, yk)
        arU, apU = auroc_auprc(Ur, yk)
        ps[k] = {"viol": float(yk.mean()), "P_mean": float(pk.mean()),
                 "AUROC_negP": ar, "AUPRC_negP": ap,
                 "AUROC_Urho": arU, "AUPRC_Urho": apU}
    s["per_spec"] = ps
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HalfCheetah-v5")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--horizons", nargs="+", type=int, default=[10])
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--extra", action="store_true", help="include compound+friction")
    ap.add_argument("--noise_abl", action="store_true")
    ap.add_argument("--m_abl", nargs="*", type=int, default=None)
    ap.add_argument("--out", default="data/results/corl_extension.json")
    a = ap.parse_args()
    envs = list(ENV_CONFIGS) if a.env == "all" else [a.env]
    conds = BASE_CONDS + (EXT_CONDS if a.extra else [])
    if a.conds: conds = [c for c in conds if c[0] in a.conds]
    out = {}
    for env in envs:
        dims = ENV_CONFIGS[env]
        for seed in a.seeds:
            for (cn, ck) in conds:
                for K in a.horizons:
                    modes = [0.0]
                    if a.noise_abl and ck.get("noise_std", 0):
                        modes = [0.0, ck["noise_std"]]
                    for rn in modes:
                        ms = a.m_abl or [3]
                        for m in ms:
                            tag = f"s{seed:03d}_K{K}_m{m}_rn{rn}"
                            print(f"{env} {cn} {tag} ...", flush=True)
                            t0 = time.time()
                            W = run_one(env, dims, seed, K, cn, ck, a.episodes, 100, rn, m)
                            s = summarize(W)
                            out.setdefault(env, {}).setdefault(cn, {})[tag] = s
                            print(f"  n={s.get('n')} viol={s.get('viol',0):.1%} AUROC_Urho={s.get('AUROC_U_rho',s.get('AUROC_U_rho'))} t={time.time()-t0:.0f}s", flush=True)
    with open(f"{PROJECT_ROOT}/{a.out}", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {a.out}")

if __name__ == "__main__":
    main()
