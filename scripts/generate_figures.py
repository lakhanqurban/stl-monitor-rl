"""Generate all paper figures for SPAIS 2026.

Creates publication-quality figures:
1. Monitor comparison (nominal trajectories)
2. Distribution shift degradation curves
3. Ablation results
4. Cross-environment comparison
5. Reactive vs Predictive monitoring
6. World model training loss

Usage:
    python scripts/generate_figures.py
"""

import os, sys, warnings, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_ROOT = Path(__file__).parent.parent
FIG_DIR = PROJECT_ROOT / 'paper'
FIG_DIR.mkdir(exist_ok=True)

# Style settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'stl': '#2196F3',
    'threshold': '#F44336',
    'cbf': '#4CAF50',
    'reactive': '#2196F3',
    'predictive': '#FF9800',
    'ant': '#1976D2',
    'hc': '#F57C00',
}


def load_results():
    """Load all experimental results."""
    results = {}

    # Nominal results
    p = PROJECT_ROOT / 'data' / 'checkpoints' / 'results_v5.json'
    if p.exists():
        with open(p) as f:
            results['nominal_ant'] = json.load(f)

    p = PROJECT_ROOT / 'data' / 'checkpoints_halfcheetah' / 'results.json'
    if p.exists():
        with open(p) as f:
            results['nominal_hc'] = json.load(f)

    # Shift results
    p = PROJECT_ROOT / 'data' / 'results' / 'shift_results.json'
    if p.exists():
        with open(p) as f:
            results['shift_ant'] = json.load(f)

    p = PROJECT_ROOT / 'data' / 'results_halfcheetah' / 'shift_results.json'
    if p.exists():
        with open(p) as f:
            results['shift_hc'] = json.load(f)

    # Ablation
    p = PROJECT_ROOT / 'data' / 'results' / 'ablation_results.json'
    if p.exists():
        with open(p) as f:
            results['ablation'] = json.load(f)

    # Predictive shift
    p = PROJECT_ROOT / 'data' / 'predictive_shift_results.json'
    if p.exists():
        with open(p) as f:
            results['predictive_shift'] = json.load(f)

    # World model training
    p = PROJECT_ROOT / 'data' / 'world_model_Ant-v5.pt'
    if p.exists():
        import torch
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        results['wm_losses'] = ckpt.get('losses', None)

    return results


def fig1_monitor_comparison(results):
    """Figure 1: Monitor comparison on nominal Ant-v5 trajectories."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    seeds = ['0', '42', '123', '456', '789']
    nominal = results.get('nominal_ant', {})

    # (a) Violation rates
    ax = axes[0]
    stl_rates = [40, 40, 60, 80, 0]
    thr_rates = [6, 6, 7, 7, 0]
    x = np.arange(len(seeds))
    w = 0.35
    ax.bar(x - w/2, stl_rates, w, label='STL', color=COLORS['stl'], alpha=0.8)
    ax.bar(x + w/2, thr_rates, w, label='Threshold', color=COLORS['threshold'], alpha=0.8)
    ax.set_xlabel('Seed')
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('(a) Violation Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.set_ylim(0, 100)

    # (b) STL scores per seed
    ax = axes[1]
    stl_scores = [0.122, -0.042, 0.021, -0.041, 0.198]
    colors = [COLORS['stl'] if s > 0 else COLORS['threshold'] for s in stl_scores]
    ax.bar(x, stl_scores, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Seed')
    ax.set_ylabel('STL Score')
    ax.set_title('(b) STL Robustness by Seed')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)

    # (c) Reward vs STL correlation
    ax = axes[2]
    rewards = [-2.9, 22.5, 27.0, 77.7, 18.0]
    ax.scatter(rewards, stl_scores, c=COLORS['stl'], s=80, zorder=5)
    for i, s in enumerate(seeds):
        ax.annotate(f'{s}', (rewards[i], stl_scores[i]), fontsize=8, ha='left')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('STL Score')
    ax.set_title(r'(c) Reward vs Safety ($r=-0.65$)')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    plt.tight_layout()
    path = FIG_DIR / 'fig1_monitor_comparison.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig2_distribution_shift(results):
    """Figure 2: Distribution shift degradation curves."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Force, Noise, Mass for Ant-v5
    conditions_force = ['0', '5', '15', '30']
    stl_force = [0.027, 0.130, -0.186, -0.069]
    thr_force = [3, 1, 13, 5]

    conditions_noise = ['0', '0.01', '0.05', '0.15']
    stl_noise = [0.027, 0.007, 0.083, -0.202]
    thr_noise = [3, 2, 0, 14]

    conditions_mass = ['0.8x', '1.0x', '1.3x', '1.6x']
    stl_mass = [0.047, 0.027, -0.107, -0.0003]
    thr_mass = [5, 3, 1, 3]

    for ax, conds, stl_s, thr_v, title, xlabel in [
        (axes[0], conditions_force, stl_force, thr_force, 'External Forces', 'Force (N)'),
        (axes[1], conditions_noise, stl_noise, thr_noise, 'Sensor Noise', r'$\sigma$'),
        (axes[2], conditions_mass, stl_mass, thr_mass, 'Mass Variation', 'Scale'),
    ]:
        x = np.arange(len(conds))
        ax2 = ax.twinx()
        l1, = ax.plot(x, stl_s, 'o-', color=COLORS['stl'], linewidth=2, markersize=6, label='STL Score')
        l2, = ax2.plot(x, thr_v, 's--', color=COLORS['threshold'], linewidth=1.5, markersize=5, label='Thr. Viol%')
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(conds)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('STL Score', color=COLORS['stl'])
        ax2.set_ylabel('Thr. Violation %', color=COLORS['threshold'])
        ax.set_title(title)
        ax.legend(handles=[l1, l2], loc='upper left', fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / 'fig2_distribution_shift.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig3_ablation(results):
    """Figure 3: Ablation study results."""
    fig, ax = plt.subplots(figsize=(6, 4))

    props = ['None\n(full)', 'Roll', 'Pitch', 'No falls', 'Contact\nsafety', 'Velocity\ntracking', 'Recovery\nbound']
    det_rates = [20, 40, 40, 20, 0, 10, 20]
    fpr = [0, 30, 30, 0, 0, 0, 20]

    x = np.arange(len(props))
    w = 0.35
    bars1 = ax.bar(x - w/2, det_rates, w, label='Detection Rate', color=COLORS['stl'], alpha=0.8)
    bars2 = ax.bar(x + w/2, fpr, w, label='False Positive Rate', color=COLORS['threshold'], alpha=0.8)

    ax.set_xlabel('Dropped Property')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Ablation: Effect of Dropping STL Properties')
    ax.set_xticks(x)
    ax.set_xticklabels(props, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 50)

    plt.tight_layout()
    path = FIG_DIR / 'fig3_ablation.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig4_cross_environment(results):
    """Figure 4: Cross-environment comparison."""
    fig, ax = plt.subplots(figsize=(6, 4))

    conditions = ['Baseline', 'Force\n5N', 'Force\n15N', 'Noise\n0.05', 'Noise\n0.15', 'Mass\n0.8x', 'Mass\n1.6x']
    ant_scores = [0.027, 0.130, -0.186, 0.083, -0.202, 0.047, -0.0003]
    hc_scores = [-0.051, -0.060, -0.056, -0.055, -0.058, -0.043, -0.060]

    x = np.arange(len(conditions))
    ax.plot(x, ant_scores, 'o-', color=COLORS['ant'], linewidth=2, markersize=6, label='Ant-v5 (3D)')
    ax.plot(x, hc_scores, 's-', color=COLORS['hc'], linewidth=2, markersize=6, label='HalfCheetah-v5 (2D)')
    ax.axhline(y=0, color='red', linewidth=0.8, linestyle='--', alpha=0.5, label='Safety Boundary')

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel('STL Robustness Score')
    ax.set_title('Cross-Environment STL Degradation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / 'fig4_cross_environment.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig5_predictive_vs_reactive(results):
    """Figure 5: Predictive vs Reactive monitoring."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ps = results.get('predictive_shift', {})

    for ax, key, title in [
        (axes[0], 'ant_mass', 'Ant-v5'),
        (axes[1], 'hc_mass', 'HalfCheetah-v5'),
    ]:
        data = ps.get(key, {})
        if not data:
            continue
        severities = sorted(data.keys(), key=float)
        reactive = [data[s]['reactive'] for s in severities]
        predictive = [data[s]['predictive'] for s in severities]
        early = [data[s]['early_pct'] for s in severities]

        x = np.arange(len(severities))
        ax.plot(x, reactive, 'o-', color=COLORS['reactive'], linewidth=2, label='Reactive')
        ax.plot(x, predictive, 's--', color=COLORS['predictive'], linewidth=2, label='Predictive')
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{float(s):.1f}x' for s in severities])
        ax.set_xlabel('Mass Scale')
        ax.set_ylabel('STL Score')
        ax.set_title(f'{title}: Reactive vs Predictive')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIG_DIR / 'fig5_predictive_vs_reactive.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig6_wm_training(results):
    """Figure 6: World model training loss."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    losses = results.get('wm_losses', None)
    if losses:
        ax.plot(range(1, len(losses)+1), losses, 'o-', color=COLORS['stl'], linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('World Model Training Loss (Ant-v5)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No training data available', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    path = FIG_DIR / 'fig6_wm_training.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def main():
    print("=== Generating Paper Figures ===\n")
    results = load_results()
    print(f"Loaded results: {list(results.keys())}\n")

    fig1_monitor_comparison(results)
    fig2_distribution_shift(results)
    fig3_ablation(results)
    fig4_cross_environment(results)
    fig5_predictive_vs_reactive(results)
    fig6_wm_training(results)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == '__main__':
    main()
