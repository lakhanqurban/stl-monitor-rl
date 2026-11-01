"""Baseline safety monitors for comparison with STL monitor."""

import numpy as np


class ThresholdMonitor:
    """Simple threshold-based safety monitor.

    Flags unsafe states when any monitored variable exceeds a hard threshold.
    Supports both upper-bound checks (value > threshold) and lower-bound
    checks (value < threshold) via the `bounds` parameter.
    """

    def __init__(self, thresholds: dict[str, float], bounds: dict[str, str] = None):
        """
        Args:
            thresholds: Dict mapping variable names to threshold values.
            bounds: Dict mapping variable names to bound type:
                    "upper" - violation when value > threshold (default)
                    "lower" - violation when value < threshold
                    "both"  - violation when |value| > threshold
                    If not specified, defaults to "both" for backward compatibility.
        """
        self.thresholds = thresholds
        self.bounds = bounds or {k: "both" for k in thresholds}
        self.violation_counts = {k: 0 for k in thresholds}

    def check(self, readings: dict[str, float]) -> dict:
        """Check if any threshold is violated.

        Args:
            readings: Dict mapping variable names to current values.

        Returns:
            Dict with violation status and details.
        """
        violations = {}
        any_violated = False

        for var_name, threshold in self.thresholds.items():
            if var_name in readings:
                val = readings[var_name]
                bound_type = self.bounds.get(var_name, "both")

                if bound_type == "upper":
                    violated = val > threshold
                    margin = float(threshold - val)
                    display_val = float(val)
                elif bound_type == "lower":
                    violated = val < threshold
                    margin = float(val - threshold)
                    display_val = float(val)
                else:  # "both"
                    violated = abs(val) > threshold
                    margin = float(threshold - abs(val))
                    display_val = float(abs(val))

                violations[var_name] = {
                    "value": display_val,
                    "threshold": threshold,
                    "bound": bound_type,
                    "violated": violated,
                    "margin": float(margin),
                }
                if violated:
                    any_violated = True
                    self.violation_counts[var_name] += 1

        return {
            "safe": not any_violated,
            "violations": violations,
            "violation_count": sum(self.violation_counts.values()),
        }

    def check_trajectory(self, trajectory: dict[str, list[float]]) -> dict:
        """Check thresholds over a full trajectory.

        Returns:
            Dict with per-step results and summary statistics.
        """
        n_steps = len(next(iter(trajectory.values())))
        step_results = []

        for t in range(n_steps):
            readings = {k: v[t] for k, v in trajectory.items() if k in self.thresholds}
            step_results.append(self.check(readings))

        n_violated = sum(1 for r in step_results if not r["safe"])
        return {
            "total_steps": n_steps,
            "violated_steps": n_violated,
            "violation_rate": n_violated / n_steps if n_steps > 0 else 0.0,
            "safe": n_violated == 0,
            "step_results": step_results,
        }

    def reset(self):
        self.violation_counts = {k: 0 for k in self.thresholds}


class AnomalyDetector:
    """Simple autoencoder-based anomaly detector.

    Learns a reconstruction of normal operating data and flags
    high-reconstruction-error states as anomalies.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        threshold_percentile: float = 95.0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.threshold_percentile = threshold_percentile
        self.mean = None
        self.std = None
        self.threshold = None
        self._build_model()

    def _build_model(self):
        """Build a simple autoencoder using PyTorch."""
        import torch
        import torch.nn as nn

        self.device = torch.device("cpu")

        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim),
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        self.model = Autoencoder(self.input_dim, self.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.MSELoss()

    def fit(self, normal_data: np.ndarray, epochs: int = 50):
        """Train autoencoder on normal operating data.

        Args:
            normal_data: Array of shape (n_samples, input_dim) with normal trajectories.
            epochs: Training epochs.
        """
        import torch

        self.mean = normal_data.mean(axis=0)
        self.std = normal_data.std(axis=0) + 1e-8
        normalized = (normal_data - self.mean) / self.std

        tensor_data = torch.FloatTensor(normalized).to(self.device)

        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            reconstructed = self.model(tensor_data)
            loss = self.loss_fn(reconstructed, tensor_data)
            loss.backward()
            self.optimizer.step()

        # Compute threshold from training data reconstruction errors
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor_data)
            errors = torch.mean((tensor_data - reconstructed) ** 2, dim=1)
        self.threshold = float(np.percentile(errors.cpu().numpy(), self.threshold_percentile))

    def predict(self, observation: np.ndarray) -> dict:
        """Predict if an observation is anomalous.

        Args:
            observation: Array of shape (input_dim,).

        Returns:
            Dict with anomaly status and reconstruction error.
        """
        import torch

        if self.mean is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        normalized = (observation - self.mean) / self.std
        tensor_obs = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor_obs)
            error = torch.mean((tensor_obs - reconstructed) ** 2).item()

        return {
            "is_anomaly": error > self.threshold,
            "reconstruction_error": error,
            "threshold": self.threshold,
            "anomaly_score": error / self.threshold if self.threshold > 0 else 0.0,
        }

    def check_trajectory(self, trajectory: np.ndarray) -> dict:
        """Check a full trajectory for anomalies.

        Args:
            trajectory: Array of shape (n_steps, input_dim).

        Returns:
            Dict with per-step anomaly results and summary.
        """
        results = [self.predict(trajectory[t]) for t in range(len(trajectory))]
        n_anomalies = sum(1 for r in results if r["is_anomaly"])
        return {
            "total_steps": len(results),
            "anomaly_steps": n_anomalies,
            "anomaly_rate": n_anomalies / len(results) if results else 0.0,
            "safe": n_anomalies == 0,
            "step_results": results,
        }


class CBFMonitor:
    """Control Barrier Function-based safety monitor.

    Uses a simple CBF formulation to check if the current state
    is within the safe set defined by barrier constraints.
    """

    def __init__(self, state_dim: int, safe_region: dict = None):
        """
        Args:
            state_dim: Dimensionality of the state space.
            safe_region: Dict defining safe bounds per state dimension.
                        e.g., {"roll": (-0.5, 0.5), "height": (0.2, 1.0)}
        """
        self.state_dim = state_dim
        self.safe_region = safe_region or {}

    def compute_barrier(self, state: dict[str, float]) -> dict:
        """Compute barrier function value.

        h(x) >= 0 means the state is safe.
        """
        barrier_values = {}
        for dim_name, (lo, hi) in self.safe_region.items():
            if dim_name in state:
                val = state[dim_name]
                # h(x) = min(val - lo, hi - val) - positive when safe
                barrier_values[dim_name] = min(val - lo, hi - val)

        min_barrier = min(barrier_values.values()) if barrier_values else 0.0
        return {
            "barrier_values": barrier_values,
            "min_barrier": min_barrier,
            "safe": min_barrier >= 0,
        }

    def check(self, state: dict[str, float]) -> dict:
        """Check if the current state satisfies CBF constraints."""
        barrier_result = self.compute_barrier(state)
        return {
            "safe": barrier_result["safe"],
            "barrier_value": barrier_result["min_barrier"],
            "barrier_details": barrier_result["barrier_values"],
        }

    def check_trajectory(self, trajectory: dict[str, list[float]]) -> dict:
        """Check CBF constraints over a full trajectory."""
        n_steps = len(next(iter(trajectory.values())))
        step_results = []

        for t in range(n_steps):
            state = {k: v[t] for k, v in trajectory.items() if k in self.safe_region}
            step_results.append(self.check(state))

        n_violated = sum(1 for r in step_results if not r["safe"])
        return {
            "total_steps": n_steps,
            "violated_steps": n_violated,
            "violation_rate": n_violated / n_steps if n_steps > 0 else 0.0,
            "safe": n_violated == 0,
            "step_results": step_results,
        }
