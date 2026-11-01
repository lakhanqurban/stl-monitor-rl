"""Generate temporal robustness plots and per-property table from detailed results."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "data/results"
FIG_DIR = "paper"
os.makedirs(FIG_DIR, exist_ok=True)

with open(f"{OUTPUT_DIR}/shift_detailed_results.json") as f:
    all_results = json.load(f)

PROPERTY_NAMES = ["stability_roll", "stability_pitch", "no_falls", "contact_safety", "velocity_tracking", "recovery_bound"]
PROPERTY_LABELS = {
    "stability_roll": "Roll",
    "stability_pitch": "Pitch",
    "no_falls": "Height",
    "contact_safety": "Contact",
    "velocity_tracking": "Velocity",
    "recovery_bound": "Recovery",
}

# ─── Figure 7: Per-property nominal violation rates ───
baseline = [r for r in all_results if r["condition"] == "baseline"][0]
prop_names = [PROPERTY_LABELS[p] for p in PROPERTY_NAMES]
prop_vr = [baseline["per_property"][p]["violation_rate"] * 100 for p in PROPERTY_NAMES]

fig, ax = plt.subplots(figsize=(7, 3.5))
colors = ["#2196F3", "#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
bars = ax.bar(prop_names, prop_vr, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Nominal Violation Rate (%)")
ax.set_title("Per-Property Nominal Violation Rates (Ant-v5, seed 42)")
ax.set_ylim(0, 110)
for bar, val in zip(bars, prop_vr):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val:.0f}%", ha="center", fontsize=9)
ax.axhline(y=6, color="gray", linestyle="--", linewidth=0.8, label="Threshold monitor (6%)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_property_nominal.png", dpi=300)
plt.close()
print(f"Saved {FIG_DIR}/fig7_property_nominal.png")

# ─── Figure 8: Per-property violation rates under distribution shift ───
conditions_of_interest = ["baseline", "force_5N", "force_15N", "force_30N", "noise_0.01", "noise_0.05", "noise_0.15", "mass_0.8x", "mass_1.3x", "mass_1.6x"]
cond_labels = ["Nominal", "5N", "15N", "30N", "σ=0.01", "σ=0.05", "σ=0.15", "0.8×", "1.3×", "1.6×"]

# Only show properties that actually vary (exclude contact_safety which is always 100%)
varying_props = ["stability_roll", "stability_pitch", "no_falls", "velocity_tracking", "recovery_bound"]
varying_labels = [PROPERTY_LABELS[p] for p in varying_props]

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
family_groups = [
    ("External Forces", ["baseline", "force_5N", "force_15N", "force_30N"], ["Nominal", "5N", "15N", "30N"]),
    ("Sensor Noise", ["baseline", "noise_0.01", "noise_0.05", "noise_0.15"], ["Nominal", "σ=0.01", "σ=0.05", "σ=0.15"]),
    ("Mass Variation", ["baseline", "mass_0.8x", "mass_1.3x", "mass_1.6x"], ["Nominal", "0.8×", "1.3×", "1.6×"]),
]

markers = ["o", "s", "^", "D", "v"]
for ax, (family_name, cond_keys, cond_lbls) in zip(axes, family_groups):
    for i, pname in enumerate(varying_props):
        vals = []
        for ck in cond_keys:
            r = [x for x in all_results if x["condition"] == ck][0]
            vals.append(r["per_property"][pname]["violation_rate"] * 100)
        ax.plot(range(len(cond_lbls)), vals, marker=markers[i], label=PROPERTY_LABELS[pname], linewidth=1.5, markersize=5)
    ax.set_xticks(range(len(cond_lbls)))
    ax.set_xticklabels(cond_lbls, rotation=30, fontsize=8)
    ax.set_title(family_name, fontsize=10)
    ax.set_ylabel("Violation Rate (%)" if ax == axes[0] else "")
    ax.set_ylim(-5, 110)
    ax.grid(True, alpha=0.3)

axes[-1].legend(fontsize=7, loc="upper right")
plt.suptitle("Per-Property Violation Rates Under Distribution Shift (Ant-v5)", fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig8_property_shift.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR}/fig8_property_shift.png")

# ─── Figure 9: Temporal robustness traces ───
# Show selected properties for selected conditions
key_conditions = ["baseline", "force_5N", "force_15N", "force_30N"]
key_labels = ["Nominal", "5N", "15N", "30N"]
key_props = ["stability_roll", "no_falls", "velocity_tracking"]
key_prop_labels = [PROPERTY_LABELS[p] for p in key_props]

fig, axes = plt.subplots(len(key_props), 1, figsize=(8, 3 * len(key_props)), sharex=True)
if len(key_props) == 1:
    axes = [axes]

colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

for prop_idx, (pname, plabel) in enumerate(zip(key_props, key_prop_labels)):
    ax = axes[prop_idx]
    for cond_idx, (cond_key, cond_lbl) in enumerate(zip(key_conditions, key_labels)):
        r = [x for x in all_results if x["condition"] == cond_key][0]
        mean_trace = r["temporal_mean"].get(pname, [])
        std_trace = r["temporal_std"].get(pname, [])
        if mean_trace:
            t = np.arange(len(mean_trace))
            mean_arr = np.array(mean_trace)
            std_arr = np.array(std_trace)
            ax.plot(t, mean_arr, color=colors[cond_idx], label=cond_lbl, linewidth=1.2)
            ax.fill_between(t, mean_arr - std_arr, mean_arr + std_arr, color=colors[cond_idx], alpha=0.15)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel(f"ρ({plabel})")
    ax.set_title(f"Property: {plabel}", fontsize=10)
    if prop_idx == 0:
        ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Timestep")
plt.suptitle("Temporal Robustness Traces: Force Perturbation (Ant-v5)", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig9_temporal_force.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR}/fig9_temporal_force.png")

# Same for noise
key_conditions_noise = ["baseline", "noise_0.01", "noise_0.05", "noise_0.15"]
key_labels_noise = ["Nominal", "σ=0.01", "σ=0.05", "σ=0.15"]

fig, axes = plt.subplots(len(key_props), 1, figsize=(8, 3 * len(key_props)), sharex=True)
if len(key_props) == 1:
    axes = [axes]

for prop_idx, (pname, plabel) in enumerate(zip(key_props, key_prop_labels)):
    ax = axes[prop_idx]
    for cond_idx, (cond_key, cond_lbl) in enumerate(zip(key_conditions_noise, key_labels_noise)):
        r = [x for x in all_results if x["condition"] == cond_key][0]
        mean_trace = r["temporal_mean"].get(pname, [])
        std_trace = r["temporal_std"].get(pname, [])
        if mean_trace:
            t = np.arange(len(mean_trace))
            mean_arr = np.array(mean_trace)
            std_arr = np.array(std_trace)
            ax.plot(t, mean_arr, color=colors[cond_idx], label=cond_lbl, linewidth=1.2)
            ax.fill_between(t, mean_arr - std_arr, mean_arr + std_arr, color=colors[cond_idx], alpha=0.15)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel(f"ρ({plabel})")
    ax.set_title(f"Property: {plabel}", fontsize=10)
    if prop_idx == 0:
        ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Timestep")
plt.suptitle("Temporal Robustness Traces: Sensor Noise (Ant-v5)", fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig10_temporal_noise.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {FIG_DIR}/fig10_temporal_noise.png")

# ─── Print per-property table for paper ───
print("\n" + "=" * 80)
print("PER-PROPERTY VIOLATION RATES (for LaTeX table)")
print("=" * 80)
header = "Condition & " + " & ".join([PROPERTY_LABELS[p] for p in PROPERTY_NAMES]) + " \\\\"
print(header)
print("\\midrule")
for r in all_results:
    vals = []
    for p in PROPERTY_NAMES:
        vr = r["per_property"][p]["violation_rate"]
        if vr is not None:
            vals.append(f"{vr:.0%}")
        else:
            vals.append("N/A")
    print(f"{r['condition']} & {' & '.join(vals)} \\\\")
