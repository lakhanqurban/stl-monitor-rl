"""Train world-model ensembles for the locomotion environments."""
import argparse
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from scripts.world_model_train import WorldModel, WorldModelTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

envs = {
    "Ant-v5": {"obs_dim": 105, "action_dim": 8},
    "HalfCheetah-v5": {"obs_dim": 17, "action_dim": 6},
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=[*envs, "all"], default="all")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    selected_envs = envs.items() if args.env == "all" else [(args.env, envs[args.env])]

    for env_name, dims in selected_envs:
        print(f"\n{'='*50}")
        print(f"Training ensemble: {env_name}")
        print(f"{'='*50}")

        ensemble = []
        for i in range(args.ensemble_size):
            print(f"\n--- Member {i+1}/{args.ensemble_size} ---")
            torch.manual_seed(i * 42 + 7)

            wm = WorldModel(dims["obs_dim"], dims["action_dim"], latent_dim=128).to(device)
            trainer = WorldModelTrainer(wm, lr=1e-3)

            obs, acts, rews = trainer.collect_data(env_name, n_episodes=args.episodes)
            print(f"  Data: {obs.shape[0]} transitions")

            losses = trainer.train(obs, acts, rews, epochs=args.epochs)
            print(f"  Final loss: {losses[-1]:.4f}")

            ensemble.append(wm)

        # Save ensemble
        save_path = PROJECT_ROOT / "data" / f"ensemble_{env_name}.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            [{"model_state_dict": wm.state_dict()} for wm in ensemble],
            save_path,
        )
        print(f"\nSaved ensemble to {save_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
