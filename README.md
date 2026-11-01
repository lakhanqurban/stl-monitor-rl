# Beyond Ensemble Disagreement

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
[![CoRL 2026](https://img.shields.io/badge/CoRL-2026-6f42c1.svg)](https://www.corl.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stable-Baselines3 2.7+](https://img.shields.io/badge/Stable--Baselines3-2.7%2B-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium 1.0+](https://img.shields.io/badge/Gymnasium-1.0%2B-0081A5.svg)](https://gymnasium.farama.org/)
[![MuJoCo 3.1+](https://img.shields.io/badge/MuJoCo-3.1%2B-6C48A8.svg)](https://mujoco.org/)
[![RTAMT 0.3.5+](https://img.shields.io/badge/RTAMT-0.3.5%2B-555555.svg)](https://github.com/yalctse/rtamt)

Code and saved results for **“Beyond Ensemble Disagreement: Predictive STL Monitoring of RL Locomotion under Distribution Shift.”** The project evaluates whether ensemble disagreement helps assess predicted STL robustness for frozen reinforcement-learning locomotion policies under distribution shift.

The experiments use MuJoCo **Ant-v5** and **HalfCheetah-v5**. They compare predicted STL robustness with disagreement measured in STL, observation, and latent representations, using external forces, observation noise, mass changes, and selected compound and friction shifts.

## Repository contents

| Path                               | Contents                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| `src/monitors/`                  | STL monitor and comparison baselines                                           |
| `src/environments/`              | MuJoCo observation wrapper                                                     |
| `src/training/`                  | PPO training and trajectory logging                                            |
| `scripts/corl_extension_eval.py` | Disagreement, per-specification, risk-coverage, and ablation evaluation        |
| `scripts/corl_review2_eval.py`   | Current-state baseline, safe-at-start analysis, and penalized-score evaluation |
| `scripts/world_model_train.py`   | World-model architecture and training utilities                                |
| `configs/`                       | STL specifications and experiment settings                                     |
| `data/checkpoints*/`             | PPO checkpoints for the three primary seeds                                    |
| `data/ensemble_*.pt`             | Trained world-model ensembles used by the evaluations                          |
| `data/results/`                  | Saved evaluation-summary JSON files                                            |
| `tests/`                         | Smoke tests for environments, monitors, and logging                            |

The JSON files contain aggregate summaries, not raw per-window traces. The saved checkpoints and ensemble weights let you run the paper evaluation scripts without retraining.

## Install

Requirements: Python 3.11+, a MuJoCo-supported platform, and PyTorch. A CUDA-capable GPU is optional.

```bash
python -m venv .venv
```

Activate the environment, then install the project and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For CUDA, install the PyTorch build appropriate for your system as described at [pytorch.org](https://pytorch.org/).

## Run the saved-artifact evaluations

The commands below use the included PPO checkpoints and ensembles; they do not train models. Run them from the repository root.

### HalfCheetah current-state and safe-at-start analysis

```bash
python scripts/corl_review2_eval.py `
  --env HalfCheetah-v5 `
  --seeds 0 42 123 `
  --episodes 10 `
  --horizons 10 20 `
  --extra `
  --out data/results/reproduced_hc.json
```

### Ant any-violation analysis

```bash
python scripts/corl_review2_eval.py `
  --env Ant-v5 `
  --seeds 0 42 123 `
  --episodes 10 `
  --horizons 10 `
  --extra `
  --out data/results/reproduced_ant.json
```

These commands use PowerShell line continuations. In Bash, replace the backticks with `\` or put each command on one line.

### Extended evaluation and ablations

```bash
python scripts/corl_extension_eval.py `
  --env all `
  --seeds 0 42 123 `
  --episodes 10 `
  --horizons 10 20 `
  --extra `
  --out data/results/reproduced_extension.json
```

Optional flags include `--noise_abl` for the rollout-noise ablation and `--m_abl 5 10` for ensemble-size comparisons. These evaluations can take substantial time.

Each script writes a summary JSON to the path passed via `--out`. The saved files under `data/results/` record the result summaries used during development of the paper.

## Retrain the models

The repository includes the checkpoints used for evaluation. To train replacement PPO policies, run `src.training.train_baseline` once per seed and environment. For example:

```bash
python -m src.training.train_baseline --env_id Ant-v5 --seed 0 --total_timesteps 200000 --output_dir data/checkpoints
python -m src.training.train_baseline --env_id HalfCheetah-v5 --seed 0 --total_timesteps 200000 --output_dir data/checkpoints_halfcheetah
```

Repeat with seeds `42` and `123` as needed. Then train a new ensemble for one environment:

```bash
python scripts/train_ensemble.py --env Ant-v5
python scripts/train_ensemble.py --env HalfCheetah-v5
```

By default, ensemble training uses 50 nominal episodes, 30 epochs, and three members. Use `--help` to inspect the available options. Retraining produces new weights and will not necessarily reproduce the saved metrics exactly.

## Evaluation semantics

The evaluators compute STL robustness for each configured property on the available prediction window. At horizons below the recovery property's full response interval, recovery robustness is evaluated on the truncated trace. The any-violation label is the union of the per-specification violation labels. The composite score follows the aggregation implemented in the evaluator; consult `scripts/corl_extension_eval.py` and `scripts/corl_review2_eval.py` when interpreting it.

## Tests

From the repository root:

```bash
python -m pytest -q
```

## Citation

Qurban Ali, Badaruddin Chachar, and Maqsood Ali. *Beyond Ensemble Disagreement: Predictive STL Monitoring of RL Locomotion under Distribution Shift.* CoRL 2026 manuscript.(to be updated).

## License

MIT License; see [LICENSE](LICENSE).
