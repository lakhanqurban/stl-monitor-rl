"""DreamerV3-style World Model for Predictive STL Monitoring.

Trains a latent dynamics model on MuJoCo locomotion data,
then uses it to predict future trajectories for STL evaluation.

Usage:
    python scripts/world_model_train.py --env Ant-v5 --episodes 100
    python scripts/predictive_stl_monitor.py --env Ant-v5 --prediction_horizon 50
"""

import os
import sys
import warnings
import numpy as np
import yaml
from pathlib import Path
from collections import deque

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Encoder(nn.Module):
    """Encodes observations into latent state."""

    def __init__(self, obs_dim, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class Decoder(nn.Module):
    """Decodes latent state back to observations."""

    def __init__(self, latent_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, obs_dim),
        )

    def forward(self, latent):
        return self.net(latent)


class DynamicsModel(nn.Module):
    """Predicts next latent state given current latent state and action."""

    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        next_latent = self.net(x)
        reward = self.reward_head(next_latent)
        return next_latent, reward


class WorldModel(nn.Module):
    """DreamerV3-style world model for MuJoCo locomotion."""

    def __init__(self, obs_dim, action_dim, latent_dim=128):
        super().__init__()
        self.encoder = Encoder(obs_dim, latent_dim)
        self.decoder = Decoder(latent_dim, obs_dim)
        self.dynamics = DynamicsModel(latent_dim, action_dim, hidden_dim=256)
        self.latent_dim = latent_dim

    def forward(self, obs, action):
        """Single-step forward pass."""
        latent = self.encoder(obs)
        next_latent, reward_pred = self.dynamics(latent, action)
        obs_recon = self.decoder(next_latent)
        return next_latent, obs_recon, reward_pred

    def predict_trajectory(self, initial_obs, actions, horizon):
        """Predict future trajectory given initial observation and action sequence.
        
        Args:
            initial_obs: (batch, obs_dim) initial observation
            actions: (batch, horizon, action_dim) action sequence
            horizon: int, prediction horizon
            
        Returns:
            predicted_obs: (batch, horizon, obs_dim) predicted observations
            predicted_rewards: (batch, horizon, 1) predicted rewards
        """
        batch_size = initial_obs.shape[0]
        device = initial_obs.device
        
        latent = self.encoder(initial_obs)
        predicted_obs = []
        predicted_rewards = []
        
        for t in range(horizon):
            action = actions[:, t]
            latent, reward = self.dynamics(latent, action)
            obs = self.decoder(latent)
            predicted_obs.append(obs)
            predicted_rewards.append(reward)
        
        predicted_obs = torch.stack(predicted_obs, dim=1)
        predicted_rewards = torch.stack(predicted_rewards, dim=1)
        
        return predicted_obs, predicted_rewards


class WorldModelTrainer:
    """Trains the world model on MuJoCo trajectory data."""

    def __init__(self, world_model, lr=1e-3):
        self.model = world_model
        self.optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)
        self.device = next(world_model.parameters()).device

    def collect_data(self, env_name, n_episodes=100, max_steps=500):
        """Collect trajectory data from MuJoCo environment."""
        import gymnasium as gym
        from stable_baselines3 import PPO
        
        # Load the checkpoint trained for this environment.
        checkpoint_dir = (
            "checkpoints_halfcheetah" if env_name == "HalfCheetah-v5" else "checkpoints"
        )
        model_path = PROJECT_ROOT / "data" / checkpoint_dir / "seed_000" / "final_model"
        env = gym.make(env_name)
        
        if model_path.exists():
            model = PPO.load(str(model_path))
        else:
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=10000)
        
        observations = []
        actions = []
        rewards = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            for t in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                
                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                
                obs = next_obs
                if terminated or truncated:
                    break
        
        env.close()
        
        obs_tensor = torch.FloatTensor(np.array(observations)).to(self.device)
        act_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        rew_tensor = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        
        return obs_tensor, act_tensor, rew_tensor

    def train(self, observations, actions, rewards, epochs=50, batch_size=256):
        """Train world model on collected data."""
        dataset = torch.utils.data.TensorDataset(observations[:-1], actions[:-1], 
                                                   observations[1:], rewards[:-1])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for obs_batch, act_batch, next_obs_batch, rew_batch in dataloader:
                self.optimizer.zero_grad()
                
                next_latent_pred, _, rew_pred = self.model(obs_batch, act_batch)
                next_obs_pred = self.model.decoder(next_latent_pred)
                
                obs_loss = F.mse_loss(next_obs_pred, next_obs_batch)
                rew_loss = F.mse_loss(rew_pred, rew_batch)
                loss = obs_loss + 0.1 * rew_loss
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        
        return losses


class PredictiveSTLMonitor:
    """Evaluates STL on predicted future trajectories from world model."""

    def __init__(self, world_model, stl_monitor, device='cpu'):
        self.world_model = world_model
        self.stl_monitor = stl_monitor
        self.device = device

    def predict_and_evaluate(self, current_obs, action_sequence, horizon=50):
        """Predict future trajectory and evaluate STL on predictions.
        
        Args:
            current_obs: (1, obs_dim) current observation
            action_sequence: (1, horizon, action_dim) predicted actions
            horizon: prediction horizon in timesteps
            
        Returns:
            dict with reactive_score, predictive_score, and per-property scores
        """
        self.world_model.eval()
        
        with torch.no_grad():
            predicted_obs, predicted_rewards = self.world_model.predict_trajectory(
                current_obs, action_sequence, horizon
            )
        
        # Extract monitoring signals from predicted observations
        # This mapping depends on the environment
        predicted_signals = self._extract_signals(predicted_obs)
        
        # Evaluate STL on predicted trajectory
        predictive_scores = {}
        for prop_name in self.stl_monitor.monitors:
            result = self.stl_monitor.evaluate_online_step(
                prop_name, 0, predicted_signals
            )
            predictive_scores[prop_name] = result['robustness']
        
        predictive_composite = self.stl_monitor.get_safety_score(
            {k: {'robustness': v} for k, v in predictive_scores.items()}
        )
        
        return {
            'predictive_score': predictive_composite,
            'predictive_scores': predictive_scores,
            'predicted_obs': predicted_obs.cpu().numpy(),
        }

    def _extract_signals(self, predicted_obs):
        """Extract monitoring signals from predicted observations."""
        obs_np = predicted_obs.cpu().numpy()
        if obs_np.ndim == 3:
            obs_np = obs_np[0]  # (horizon, obs_dim) -> take first timestep
        
        return {
            'body_height': float(obs_np[0, 2]) if obs_np.shape[1] > 2 else 0.65,
            'body_roll': float(obs_np[0, 4]) if obs_np.shape[1] > 4 else 0.0,
            'body_pitch': float(obs_np[0, 5]) if obs_np.shape[1] > 5 else 0.0,
            'body_velocity_x': float(obs_np[0, 0]) if obs_np.shape[1] > 0 else 0.0,
            'n_airborne_feet': 2.0,
        }


def main():
    """Train world model and demonstrate predictive STL monitoring."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Ant-v5')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--prediction_horizon', type=int, default=20)
    args = parser.parse_args()
    
    print(f"=== World Model Training: {args.env} ===")
    
    # Determine dimensions
    if args.env == 'Ant-v5':
        obs_dim, action_dim = 105, 8
    else:
        obs_dim, action_dim = 17, 6
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create and train world model
    world_model = WorldModel(obs_dim, action_dim, latent_dim=128).to(device)
    trainer = WorldModelTrainer(world_model, lr=1e-3)
    
    print(f"\nCollecting data ({args.episodes} episodes)...")
    obs, acts, rews = trainer.collect_data(args.env, n_episodes=args.episodes)
    print(f"  Data: {obs.shape[0]} transitions, obs_dim={obs_dim}, act_dim={action_dim}")
    
    print(f"\nTraining world model ({args.epochs} epochs)...")
    losses = trainer.train(obs, acts, rews, epochs=args.epochs)
    print(f"  Final loss: {losses[-1]:.4f}")
    
    # Save model
    save_path = PROJECT_ROOT / 'data' / f'world_model_{args.env}.pt'
    torch.save({
        'model_state_dict': world_model.state_dict(),
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'losses': losses,
    }, save_path)
    print(f"  Model saved to {save_path}")
    
    # Demonstrate predictive monitoring
    print(f"\n=== Predictive STL Monitoring Demo ===")
    config_file = PROJECT_ROOT / 'configs' / (
        'stl_properties_halfcheetah.yaml' if args.env == 'HalfCheetah-v5' else 'stl_properties.yaml'
    )
    with open(config_file) as f:
        stl_config = yaml.safe_load(f)
    
    from src.monitors.stl_monitor import STLSafetyMonitor
    stl_monitor = STLSafetyMonitor(stl_config['properties'])
    predictive_monitor = PredictiveSTLMonitor(world_model, stl_monitor, device)
    
    # Get a sample trajectory
    import gymnasium as gym
    from stable_baselines3 import PPO
    
    env = gym.make(args.env)
    checkpoint_dir = (
        "checkpoints_halfcheetah" if args.env == "HalfCheetah-v5" else "checkpoints"
    )
    model_path = PROJECT_ROOT / "data" / checkpoint_dir / "seed_000" / "final_model"
    if model_path.exists():
        model = PPO.load(str(model_path))
    else:
        model = PPO("MlpPolicy", env, verbose=0)
    
    obs, _ = env.reset()
    action_seq = []
    obs_seq = [obs.copy()]
    
    for t in range(args.prediction_horizon):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        action_seq.append(action)
        obs_seq.append(obs.copy())
        if terminated or truncated:
            obs, _ = env.reset()
    
    action_tensor = torch.FloatTensor(np.array([action_seq])).to(device)
    obs_tensor = torch.FloatTensor(np.array([obs_seq[0]])).to(device)
    
    # Predictive evaluation
    result = predictive_monitor.predict_and_evaluate(
        obs_tensor, action_tensor, horizon=args.prediction_horizon
    )
    
    print(f"  Predictive STL score: {result['predictive_score']:.4f}")
    for prop, score in result['predictive_scores'].items():
        status = "OK" if score > 0 else "VIOLATION"
        print(f"    {prop}: {score:.4f} [{status}]")
    
    env.close()
    print(f"\n=== Done ===")


if __name__ == '__main__':
    main()
