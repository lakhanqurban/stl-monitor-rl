"""CoRL review-2 evaluation: current-state baseline, safe-at-start future
violations, S_UP/risk-coverage on the any-violation target, height-threshold
sensitivity.

Extends corl_extension_eval.py (same rollout protocol: 100-step episodes,
stride q=max(1,K), M=3, seeds 0/42/123) with per-window records:
  O_inst      composite instantaneous robustness at time t (pointwise specs)
  Y_inst_any  any pointwise spec violated at time t (safe-at-start mask)
  Y_future_any  any spec violated on t+1..t+K (future-only label)
Usage:
  python scripts/corl_review2_eval.py --env HalfCheetah-v5 --seeds 0 42 123 \
    --episodes 10 --horizons 10 --conds baseline noise_0.15 \
    --out data/results/corl_review2_hc.json
  python scripts/corl_review2_eval.py --h_min_override no_falls=0.08 ...
  python scripts/corl_review2_eval.py --env Ant-v5 ...
"""
import os, sys, json, time, argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import numpy as np
import torch
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from src.monitors.stl_monitor import STLSafetyMonitor
from scripts.world_model_train import WorldModel
from corl_extension_eval import (
    ENV_CONFIGS, BASE_CONDS, EXT_CONDS, eval_stl_perspec,
    closed_loop_predict, auroc_auprc, risk_coverage, device, PROJECT_ROOT,
)

# specs with no meaning at a single instant (response properties)
POINTWISE_EXCLUDE = {"recovery_bound"}


def run_one_ext(env_name, dims, seed, K, cond_name, ck, n_ep, max_steps, m_sub=3,
                h_min_override=None, rollout_noise=0.0):
    np.random.seed(2026)
    cand = [f"{PROJECT_ROOT}/data/ensemble_{env_name}_M10.pt",
            f"{PROJECT_ROOT}/data/ensemble_{env_name}_M5.pt",
            f"{PROJECT_ROOT}/data/ensemble_{env_name}.pt"]
    src = next(p for p in cand if os.path.exists(p))
    saved = torch.load(src, map_location=device)
    ens = []
    for item in saved[:m_sub]:
        wm = WorldModel(dims["obs_dim"], dims["action_dim"], latent_dim=128).to(device)
        wm.load_state_dict(item["model_state_dict"]); wm.eval(); ens.append(wm)
    pol = PPO.load(f"{PROJECT_ROOT}/data/{dims['ckpt_dir']}/seed_{seed:03d}/final_model")
    with open(f"{PROJECT_ROOT}/configs/{dims['config']}") as f:
        props = yaml.safe_load(f)["properties"]
    if h_min_override:
        for spec_name, val in h_min_override.items():
            props[spec_name]["parameters"]["h_min"] = val
    stl = STLSafetyMonitor(props)
    fm = ck.get("force_mag", 0.0); ns = ck.get("noise_std", 0.0)
    ms = ck.get("mass_scale", 1.0); fs = ck.get("friction_scale", 1.0)
    W = []
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
                try: env.unwrapped.data.xfrc_applied[1, 0] = float(fm)
                except Exception: pass
            obs, _, term, trunc, _ = env.step(a)
            try: env.unwrapped.data.xfrc_applied[:] = 0
            except Exception: pass
            if ns > 0: obs = obs + np.random.normal(0, ns, obs.shape).astype(obs.dtype)
            ep_obs.append(obs.copy())
            if term or trunc: break
        env.close()
        ep_obs = np.array(ep_obs); T = len(ep_obs) - 1
        for t in range(0, min(T - K, max_steps), max(1, K)):
            o_scores = eval_stl_perspec(stl, ep_obs[t:t+K+1], env_name)
            o_comp = float(np.mean(list(o_scores.values())))
            preds, _lats = closed_loop_predict(ens, pol, ep_obs[t], K, rollout_noise)
            m_scores = [eval_stl_perspec(stl, preds[m], env_name) for m in range(len(ens))]
            keys = list(o_scores.keys())
            p_mat = np.array([[s[k] for k in keys] for s in m_scores])
            comp_m = p_mat.mean(1); u_rho = float(comp_m.std())
            p_comp = float(comp_m.mean())
            pm = preds[:, 1:, :].mean(0)
            actual = ep_obs[t+1:t+K+1]
            e_l2 = float(np.mean(np.linalg.norm(pm - actual, axis=1))) if len(actual) == K else float("nan")
            stl_err = abs(p_comp - o_comp)
            y_comp = int(o_comp < 0)
            y_ps = {k: int(v < 0) for k, v in o_scores.items()}
            p_ps = {k: float(v) for k, v in zip(keys, p_mat.mean(0))}
            # instantaneous (current-state) robustness at time t
            o_inst_ps = eval_stl_perspec(stl, [ep_obs[t]], env_name)
            o_inst_comp = float(np.mean(
                [v for k, v in o_inst_ps.items() if k not in POINTWISE_EXCLUDE]))
            y_inst_ps = {k: int(v < 0) for k, v in o_inst_ps.items()
                         if k not in POINTWISE_EXCLUDE}
            y_inst_any = int(any(y_inst_ps.values()))
            # future-only label: violation on t+1..t+K
            f_scores = eval_stl_perspec(stl, ep_obs[t+1:t+K+1], env_name)
            y_future_any = int(any(v < 0 for v in f_scores.values()))
            W.append({"O": o_comp, "P": p_comp, "U_rho": u_rho,
                      "E_l2": e_l2, "STL_err": stl_err, "Y": y_comp,
                      "O_ps": o_scores, "P_ps": p_ps, "Y_ps": y_ps,
                      "O_inst": o_inst_comp, "Y_inst_any": y_inst_any,
                      "Y_future_any": y_future_any})
    return W


def summarize_ext(W):
    if not W:
        return {"n": 0}
    O = np.array([w["O"] for w in W]); P = np.array([w["P"] for w in W])
    Ur = np.array([w["U_rho"] for w in W]); SE = np.array([w["STL_err"] for w in W])
    Y = np.array([w["Y"] for w in W], int)
    Ya = np.array([int(any(w["Y_ps"].values())) for w in W], int)
    Oi = np.array([w["O_inst"] for w in W])
    Yi = np.array([w["Y_inst_any"] for w in W], int)
    Yf = np.array([w["Y_future_any"] for w in W], int)
    s = {"n": len(W), "viol": float(Y.mean()), "viol_any": float(Ya.mean()),
         "inst_any_rate": float(Yi.mean()),
         "O_mean": float(O.mean()), "P_mean": float(P.mean()),
         "bias": float((P - O).mean()), "STL_err": float(SE.mean())}
    # headline metrics on both targets (reproduce paper + new)
    for tag, lab in [("Y", Y), ("ANY", Ya)]:
        for nm, sc in [("U_rho", Ur), ("negP", -P)]:
            ar, ap = auroc_auprc(sc, lab)
            s[f"{tag}_AUROC_{nm}"] = ar; s[f"{tag}_AUPRC_{nm}"] = ap
    # current-state baseline: -O_inst vs any-violation target
    ar, ap = auroc_auprc(-Oi, Ya)
    s["INST_AUROC_negO"] = ar; s["INST_AUPRC_negO"] = ap
    ar, ap = auroc_auprc(-Oi, Yf)
    s["INST_AUROC_negO_future"] = ar
    # safe-at-start subset: future violations only
    mask = Yi == 0
    s["safe_n"] = int(mask.sum()); s["safe_frac"] = float(mask.mean())
    s["safe_viol_future"] = float(Yf[mask].mean()) if mask.any() else float("nan")
    for nm, sc in [("U_rho", Ur), ("negP", -P), ("negOinst", -Oi)]:
        ar, ap = auroc_auprc(sc[mask], Yf[mask]) if mask.any() else (float("nan"),) * 2
        s[f"SAFE_AUROC_{nm}"] = ar; s[f"SAFE_AUPRC_{nm}"] = ap
    # S_UP sweep on the any-violation target
    for lam in [0, 1, 2, 5]:
        SUP = P - lam * Ur
        ar, ap = auroc_auprc(-SUP, Ya)
        s[f"ANY_AUROC_SUP{lam}"] = ar; s[f"ANY_AUPRC_SUP{lam}"] = ap
        s[f"ANY_SUP{lam}_flagrate"] = float((SUP < 0).mean())
    # risk-coverage on the any-violation target
    s["risk_coverage_Urho_ANY"] = risk_coverage(Ur, SE, Ya)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HalfCheetah-v5")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 123])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--horizons", nargs="+", type=int, default=[10])
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--extra", action="store_true")
    ap.add_argument("--h_min_override", default=None,
                    help="e.g. no_falls=0.08")
    ap.add_argument("--out", default="data/results/corl_review2.json")
    a = ap.parse_args()
    dims = ENV_CONFIGS[a.env]
    conds = BASE_CONDS + (EXT_CONDS if a.extra else [])
    if a.conds:
        conds = [c for c in conds if c[0] in a.conds]
    hmo = None
    if a.h_min_override:
        k, v = a.h_min_override.split("=")
        hmo = {k: float(v)}
    out = {}
    for seed in a.seeds:
        for (cn, ck) in conds:
            for K in a.horizons:
                tag = f"s{seed:03d}_K{K}_m3"
                print(f"{a.env} {cn} {tag} ...", flush=True)
                t0 = time.time()
                W = run_one_ext(a.env, dims, seed, K, cn, ck, a.episodes, 100,
                                h_min_override=hmo)
                s = summarize_ext(W)
                out.setdefault(cn, {})[tag] = s
                print(f"  n={s['n']} viol_any={s['viol_any']:.1%} "
                      f"ANY_AUROC_negP={s.get('ANY_AUROC_negP', float('nan')):.3f} "
                      f"INST_AUROC={s.get('INST_AUROC_negO', float('nan')):.3f} "
                      f"safe_frac={s['safe_frac']:.1%} "
                      f"safe_viol={s.get('safe_viol_future', float('nan')):.1%} "
                      f"t={time.time()-t0:.0f}s", flush=True)
    with open(f"{PROJECT_ROOT}/{a.out}", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
