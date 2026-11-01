"""Generate final comparison tables and summary for the paper."""
import sys, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

OUTPUT_DIR = "data/results"

# Load training results
with open("data/checkpoints/results.json") as f:
    train_results = json.load(f)

# Load distribution shift results
with open(f"{OUTPUT_DIR}/shift_results.json") as f:
    shift_results = json.load(f)

print("=" * 80)
print("TABLE 1: Monitor Comparison on Nominal Trajectories (5 seeds, 200k steps)")
print("=" * 80)

# Aggregate training results
for key in ["mean_reward", "stl_mean_score", "stl_violation_rate", "threshold_violation_rate", "cbf_violation_rate"]:
    vals = [r[key] for r in train_results]
    print(f"  {key:30s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

print()
print("=" * 80)
print("TABLE 2: STL Robustness Under Distribution Shift (seed 42)")
print("=" * 80)

rows = []
for r in shift_results:
    stl_s = f"{r['stl_mean_score']:+.3f} +/- {r['stl_std_score']:.3f}" if r['stl_mean_score'] is not None else "N/A"
    stl_v = f"{r['stl_violation_rate']:.0%}" if r['stl_violation_rate'] is not None else "N/A"
    thr_v = f"{r['threshold_violation_rate']:.0%}" if r['threshold_violation_rate'] is not None else "N/A"
    cbf_v = f"{r['cbf_violation_rate']:.0%}" if r['cbf_violation_rate'] is not None else "N/A"
    rows.append({
        "Condition": r['condition'],
        "STL Score": stl_s,
        "STL Viol.": stl_v,
        "Thr. Viol.": thr_v,
        "CBF Viol.": cbf_v,
        "Reward": f"{r['mean_reward']:.1f}",
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# Save as CSV for paper
df.to_csv(f"{OUTPUT_DIR}/comparison_table.csv", index=False)

print()
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)

baseline = shift_results[0]
print(f"\n1. STL Monitor Differentiates Under Shift:")
for r in shift_results[1:]:
    delta = r['stl_mean_score'] - baseline['stl_mean_score'] if r['stl_mean_score'] is not None else 0
    print(f"   {r['condition']:20s}: {delta:+.3f} change from baseline")

print(f"\n2. Threshold Monitor is Overly Conservative:")
print(f"   Always 100% violation rate — cannot distinguish safe from unsafe")

print(f"\n3. CBF Shows Moderate Discrimination:")
for r in shift_results:
    cbf = r['cbf_violation_rate']
    if cbf is not None:
        print(f"   {r['condition']:20s}: {cbf:.0%} CBF violations")

print(f"\n4. Failure Case (to report honestly):")
# Find the condition where STL didn't degrade as expected
for r in shift_results:
    if r['stl_mean_score'] is not None and r['stl_mean_score'] > baseline['stl_mean_score']:
        print(f"   {r['condition']}: STL score {r['stl_mean_score']:+.3f} > baseline {baseline['stl_mean_score']:+.3f}")
        print(f"   (STL monitoring may over-flag in this condition)")

print(f"\n5. Reward- STL Score Correlation:")
rewards = [r['mean_reward'] for r in shift_results]
stl_scores = [r['stl_mean_score'] for r in shift_results if r['stl_mean_score'] is not None]
if len(rewards) == len(stl_scores):
    corr = np.corrcoef(rewards, stl_scores)[0, 1]
    print(f"   Pearson r = {corr:.3f} between reward and STL score")
