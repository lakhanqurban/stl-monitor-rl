# Predictive STL Monitoring of RL Locomotion

Research code accompanying **“Beyond Ensemble Disagreement: Predictive STL Monitoring of RL Locomotion under Distribution Shift.”** The project evaluates closed-loop, world-model-based Signal Temporal Logic (STL) monitoring on MuJoCo Ant-v5 and HalfCheetah-v5.

The experiments compare predicted STL robustness with STL-, observation-, and latent-space ensemble disagreement under external forces, observation noise, mass changes, and other shifts. The primary analysis uses PPO seeds `0`, `42`, and `123`; the legacy evaluation scripts retain their separate single-seed protocol.

## Requirements and installation

- Python 3.11 or later
- MuJoCo-supported operating system
- CUDA is optional; CPU execution is supported but world-model training is slower

```bash
git clone <your-repository-url>
cd stl-monitoring-main
python -m venv .venv
# Activate the environment, then:
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For CUDA-specific PyTorch builds, install the appropriate PyTorch wheel for your system before installing this project.

## Reproducing the evaluations

The paper's evaluation scripts are `scripts/corl_extension_eval.py` and `scripts/corl_review2_eval.py`. The PPO checkpoints, world-model ensembles, and saved result summaries used for the paper are included under `data/`.

Expected artifact paths include `data/checkpoints/seed_<seed>/final_model.zip`, `data/checkpoints_halfcheetah/seed_<seed>/final_model.zip`, and `data/ensemble_<environment>.pt`. The curated paper checkpoints and ensembles are included; generated trajectories, logs, and non-paper seed outputs are ignored by Git.

Train PPO policies for the three primary seeds (repeat for both environment IDs):

```bash
python -m src.training.train_baseline --env_id Ant-v5 --seed 0 --total_timesteps 200000 --output_dir data/checkpoints
python -m src.training.train_baseline --env_id Ant-v5 --seed 42 --total_timesteps 200000 --output_dir data/checkpoints
python -m src.training.train_baseline --env_id Ant-v5 --seed 123 --total_timesteps 200000 --output_dir data/checkpoints

python -m src.training.train_baseline --env_id HalfCheetah-v5 --seed 0 --total_timesteps 200000 --output_dir data/checkpoints_halfcheetah
python -m src.training.train_baseline --env_id HalfCheetah-v5 --seed 42 --total_timesteps 200000 --output_dir data/checkpoints_halfcheetah
python -m src.training.train_baseline --env_id HalfCheetah-v5 --seed 123 --total_timesteps 200000 --output_dir data/checkpoints_halfcheetah
```

Then train the world-model ensemble and run the primary analysis:

```bash
python scripts/train_ensemble.py --env Ant-v5
python scripts/train_ensemble.py --env HalfCheetah-v5

python scripts/corl_review2_eval.py --env HalfCheetah-v5 --seeds 0 42 123 --episodes 10 --horizons 10 20 --extra --out data/results/corl_review2_hc.json
python scripts/corl_review2_eval.py --env Ant-v5 --seeds 0 42 123 --episodes 10 --horizons 10 --extra --out data/results/corl_review2_ant.json
```

The extension evaluator provides the additional disagreement baselines and ablations:

```bash
python scripts/corl_extension_eval.py --env all --seeds 0 42 123 --episodes 10 --horizons 10 20 --extra --out data/results/corl_extension.json
```

These runs are computationally intensive. The output summaries are written under `data/results/`. The included checkpoints allow evaluation without retraining; if you retrain, generated checkpoint directories and trajectories are excluded by `.gitignore`.

### Release check

Before describing this repository as an exact reproduction package, reconcile recovery handling in the composite score: the current evaluators average all configured specification scores into the composite, including truncated recovery at $K<50$, while the manuscript describes recovery as excluded from that composite. Changing this behavior may change reported results and should be paired with a rerun or a corresponding manuscript correction.

## Tests

Run the smoke tests from the repository root:

```bash
python -m pytest -q
```

## Repository layout

```text
configs/       STL properties, training, and perturbation settings
scripts/       World-model training, paper evaluation, and analysis scripts
src/           Environment wrappers, STL monitors, training, and evaluation code
tests/         Smoke tests
data/          Locally generated checkpoints, ensembles, and evaluation outputs
```

## Scope

This repository supports the paper's simulated MuJoCo experiments. It does not provide formal safety guarantees, and its STL specifications are not claims about general physical safety. See the paper for the full protocol, interpretation, and limitations.

## License

Released under the MIT License; see [LICENSE](LICENSE).
