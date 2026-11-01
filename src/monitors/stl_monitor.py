"""STL Safety Monitor using RTAMT.

Provides both offline (trajectory-level) and online (per-timestep)
robustness evaluation for locomotion safety properties.
"""

import numpy as np
import rtamt


class STLSafetyMonitor:
    """STL-based safety monitor for locomotion policies.

    Evaluates Signal Temporal Logic properties over observed trajectories
    and computes robustness values indicating safety margins.
    """

    def __init__(self, properties_config: dict):
        self.properties_config = properties_config
        self.monitors = {}
        self._build_monitors()

    def _build_monitors(self):
        """Build RTAMT monitors for each configured STL property."""
        for prop_name, prop_config in self.properties_config.items():
            self.monitors[prop_name] = self._create_monitor(prop_config)

    def _create_monitor(self, prop_config: dict):
        """Create an RTAMT STL monitor from a property configuration."""
        spec = rtamt.StlDiscreteTimeSpecification()
        spec.name = prop_config["name"]

        # Declare variables
        for var_name, var_type in prop_config.get("variables", {}).items():
            spec.declare_var(var_name, "float")

        # Set the STL formula with parameters substituted
        formula = prop_config["formula"]
        params = prop_config.get("parameters", {})
        for param_name, param_value in params.items():
            formula = formula.replace(f"{{{param_name}}}", str(param_value))
        spec.spec = formula

        try:
            spec.parse()
            spec.pastify()
        except rtamt.RTAMTException as err:
            raise ValueError(
                f"Failed to parse STL spec '{prop_config['name']}': {err}\n"
                f"Formula after substitution: {formula}"
            )

        return {
            "spec": spec,
            "config": prop_config,
            "params": prop_config.get("parameters", {}),
        }

    def evaluate_trajectory(
        self, monitor_name: str, signal_data: dict[str, list[float]]
    ) -> dict:
        """Offline evaluation: compute robustness over a complete trajectory.

        Uses the online update() loop per timestep since RTAMT's evaluate()
        API has inconsistent behavior across versions.

        Args:
            monitor_name: Name of the property to evaluate.
            signal_data: Dict mapping variable names to lists of values
                        (one value per timestep).

        Returns:
            Dict with final robustness value and per-step trace.
        """
        monitor = self.monitors[monitor_name]

        # Create a fresh monitor instance
        fresh_spec = rtamt.StlDiscreteTimeSpecification()
        fresh_spec.name = monitor["config"]["name"]

        for var_name in monitor["config"].get("variables", {}).keys():
            fresh_spec.declare_var(var_name, "float")

        formula = monitor["config"]["formula"]
        params = monitor["config"].get("parameters", {})
        for param_name, param_value in params.items():
            formula = formula.replace(f"{{{param_name}}}", str(param_value))
        fresh_spec.spec = formula
        fresh_spec.parse()
        fresh_spec.pastify()

        # Map observation names to RTAMT variable names
        var_mapping = monitor["config"].get("variables", {})
        inverse_mapping = {v: k for k, v in var_mapping.items()}

        # Get signal length
        n_steps = 0
        for obs_name, values in signal_data.items():
            n_steps = max(n_steps, len(values))

        robustness_trace = []
        for t in range(n_steps):
            signal = []
            for obs_name, values in signal_data.items():
                stl_var_name = inverse_mapping.get(obs_name, obs_name)
                if stl_var_name in var_mapping and t < len(values):
                    signal.append((stl_var_name, float(values[t])))
            try:
                rob = fresh_spec.update(t, signal)
                robustness_trace.append(rob)
            except rtamt.RTAMTException:
                robustness_trace.append(0.0)

        final_robustness = robustness_trace[-1] if robustness_trace else 0.0

        return {
            "property": monitor_name,
            "robustness": final_robustness,
            "satisfied": final_robustness >= 0,
            "margin": abs(final_robustness),
            "trace": robustness_trace,
        }

    def evaluate_online_step(
        self, monitor_name: str, timestep: int, readings: dict[str, float]
    ) -> dict:
        """Online evaluation: compute robustness at a single timestep.

        Args:
            monitor_name: Name of the property to evaluate.
            timestep: Current timestep index.
            readings: Dict mapping variable names to their current values.

        Returns:
            Dict with current robustness value.
        """
        monitor = self.monitors[monitor_name]
        spec = monitor["spec"]

        # Map observation names to RTAMT variable names
        var_mapping = monitor["config"].get("variables", {})
        # var_mapping is like {"height": "body_height"}
        # We need to invert it: {"body_height": "height"}
        inverse_mapping = {v: k for k, v in var_mapping.items()}

        signal = []
        for obs_name, value in readings.items():
            stl_var_name = inverse_mapping.get(obs_name, obs_name)
            signal.append((stl_var_name, value))

        try:
            robustness = spec.update(timestep, signal)
        except rtamt.RTAMTException:
            robustness = 0.0

        return {
            "property": monitor_name,
            "timestep": timestep,
            "robustness": robustness,
            "satisfied": robustness >= 0,
            "margin": abs(robustness),
        }

    def evaluate_all_trajectory(
        self, signal_data: dict[str, list[float]]
    ) -> dict[str, dict]:
        """Evaluate all properties over a complete trajectory.

        Args:
            signal_data: Dict mapping variable names to lists of values.

        Returns:
            Dict mapping property names to evaluation results.
        """
        results = {}
        for prop_name in self.monitors:
            try:
                results[prop_name] = self.evaluate_trajectory(prop_name, signal_data)
            except Exception as e:
                results[prop_name] = {
                    "property": prop_name,
                    "robustness": None,
                    "satisfied": False,
                    "error": str(e),
                }
        return results

    def get_safety_score(self, results: dict[str, dict]) -> float:
        """Compute a composite safety score from individual property results.

        Weighted average of robustness values. Negative score means
        at least one property is violated on average.
        """
        scores = []
        weights = []
        for prop_name, result in results.items():
            if result.get("robustness") is not None:
                scores.append(result["robustness"])
                weights.append(
                    self.monitors[prop_name]["config"].get("weight", 1.0)
                )

        if not scores:
            return -1.0

        weights = np.array(weights)
        scores = np.array(scores)
        return float(np.average(scores, weights=weights))
