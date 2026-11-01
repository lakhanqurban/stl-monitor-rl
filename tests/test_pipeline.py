"""Quick smoke test to verify the full pipeline works end-to-end."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_mujoco_env():
    """Test MuJoCo environment creation and stepping."""
    import gymnasium as gym
    env = gym.make("Ant-v5", render_mode=None)
    obs, info = env.reset()
    assert obs.shape == (105,), f"Expected (105,) got {obs.shape}"
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    env.close()
    print("[PASS] MuJoCo environment")


def test_locomotion_wrapper():
    """Test LocomotionWrapper extraction of monitor observations."""
    import gymnasium as gym
    from src.environments.locomotion_wrapper import LocomotionWrapper
    env = gym.make("Ant-v5", render_mode=None)
    wrapped = LocomotionWrapper(env)
    obs, info = wrapped.reset()
    assert "monitor_obs" in info
    mon_obs = info["monitor_obs"]
    assert "body_roll" in mon_obs
    assert "body_pitch" in mon_obs
    assert "body_height" in mon_obs
    assert "body_velocity_x" in mon_obs
    assert "n_airborne_feet" in mon_obs
    action = wrapped.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped.step(action)
    assert "monitor_obs" in info
    wrapped.close()
    print("[PASS] LocomotionWrapper")


def test_stl_monitor():
    """Test STL monitor with RTAMT."""
    import yaml
    from src.monitors.stl_monitor import STLSafetyMonitor
    with open("configs/stl_properties.yaml") as f:
        config = yaml.safe_load(f)
    monitor = STLSafetyMonitor(config["properties"])
    assert "stability_roll" in monitor.monitors
    assert "no_falls" in monitor.monitors
    # Test online step
    result = monitor.evaluate_online_step(
        "no_falls", 0,
        {"body_height": 0.5}
    )
    assert result["satisfied"] is True
    result = monitor.evaluate_online_step(
        "no_falls", 1,
        {"body_height": 0.1}
    )
    assert result["satisfied"] is False
    print("[PASS] STL Monitor")


def test_baseline_monitors():
    """Test threshold and CBF monitors."""
    from src.monitors.baselines import ThresholdMonitor, CBFMonitor
    # Threshold: flag if |value| > threshold
    tm = ThresholdMonitor({"roll": 0.5, "height": 0.3})
    result = tm.check({"roll": 0.3, "height": 0.2})  # 0.3 < 0.5, 0.2 < 0.3
    assert result["safe"] is True, f"Expected safe, got {result}"
    result = tm.check({"roll": 0.6, "height": 0.2})  # 0.6 > 0.5
    assert result["safe"] is False, f"Expected unsafe, got {result}"
    # CBF: h(x) >= 0 means safe
    cbf = CBFMonitor(3, {"roll": (-0.5, 0.5), "height": (0.2, 5.0)})
    result = cbf.check({"roll": 0.3, "height": 0.5})  # 0.3 in (-0.5,0.5), 0.5 in (0.2,5.0)
    assert result["safe"] is True, f"Expected safe, got {result}"
    result = cbf.check({"roll": 0.6, "height": 0.5})  # 0.6 > 0.5
    assert result["safe"] is False, f"Expected unsafe, got {result}"
    print("[PASS] Baseline Monitors")


def test_trajectory_logger():
    """Test trajectory logging to parquet."""
    import tempfile
    import pandas as pd
    from src.training.trajectory_logger import TrajectoryLogger
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = TrajectoryLogger(tmpdir)
        logger.reset()
        mon_obs = {
            "body_roll": [0.1],
            "body_pitch": [0.05],
            "body_height": [0.4],
            "body_velocity_x": [1.0],
            "body_velocity_y": [0.0],
            "n_airborne_feet": [0],
            "contact_forces": [10.0, 10.0, 10.0, 10.0],
        }
        for t in range(10):
            logger.log_step(t, mon_obs, 1.0, t == 9)
        filepath = logger.save(seed=0, checkpoint=0, episode=0)
        df = pd.read_parquet(filepath)
        assert len(df) == 10
        assert "body_roll" in df.columns
    print("[PASS] Trajectory Logger")


def test_stl_evaluation():
    """Test full trajectory evaluation with STL monitor."""
    import yaml
    from src.monitors.stl_monitor import STLSafetyMonitor
    from src.evaluation.evaluate_monitors import evaluate_stl_monitor
    with open("configs/stl_properties.yaml") as f:
        config = yaml.safe_load(f)
    monitor = STLSafetyMonitor(config["properties"])
    # Create a safe trajectory
    trajectory = {
        "body_roll": [0.1] * 100,
        "body_pitch": [0.05] * 100,
        "body_height": [0.4] * 100,
        "body_velocity_x": [1.0] * 100,
        "body_velocity_y": [0.0] * 100,
        "n_airborne_feet": [0] * 100,
        "contact_forces_mean": [10.0] * 100,
    }
    result = evaluate_stl_monitor(monitor, trajectory)
    assert "stl_results" in result
    assert "safety_score" in result
    print("[PASS] STL Evaluation Pipeline")


if __name__ == "__main__":
    tests = [
        test_mujoco_env,
        test_locomotion_wrapper,
        test_stl_monitor,
        test_baseline_monitors,
        test_trajectory_logger,
        test_stl_evaluation,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed > 0 else 0)
