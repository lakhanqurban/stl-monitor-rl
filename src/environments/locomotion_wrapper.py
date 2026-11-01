"""LocomotionWrapper: extracts safety-relevant observations for STL monitoring.

Supports multiple MuJoCo locomotion environments:
- Ant-v5: 3D quadruped (105-dim obs, 8-dim actions)
- HalfCheetah-v5: 2D planar (17-dim obs, 6-dim actions)

MuJoCo data attributes used:
- xpos[body_id]     : body position (3D)
- xquat[body_id]    : body quaternion (w,x,y,z)
- cvel[body_id]     : body velocity (6D: lin+ang)
- cfrc_ext[body_id] : external forces on body (6D: force+torque)
- qpos[joint_start:] : joint positions
- qvel[joint_start:] : joint velocities
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# Environment-specific configurations
ENV_CONFIGS = {
    "Ant-v5": {
        "torso_id": 1,
        "foot_geom_ids": {4, 7, 10, 3},  # ankle geoms
        "n_joints": 8,
        "joint_offset": 7,  # qpos offset for joints
        "vel_offset": 6,    # qvel offset for joints
        "is_2d": False,
        "nominal_height": 0.75,
        "height_range": (0.4, 1.0),
        "roll_range": (-1.0, 1.0),
        "pitch_range": (-1.0, 1.0),
        "velocity_range": (-4.0, 4.0),
        "max_airborne_feet": 3,
    },
    "HalfCheetah-v5": {
        "torso_id": 1,
        "foot_geom_ids": {5, 8},  # bfoot, ffoot
        "n_joints": 6,
        "joint_offset": 3,  # qpos offset for joints (after rootx, rootz, rooty)
        "vel_offset": 3,    # qvel offset for joints
        "is_2d": True,
        "nominal_height": 0.5,
        "height_range": (0.3, 0.8),
        "roll_range": None,  # 2D robot - no roll
        "pitch_range": (-0.5, 0.5),
        "velocity_range": (-10.0, 10.0),
        "max_airborne_feet": 1,  # at most 1 foot airborne (2 feet total)
    },
}


class LocomotionWrapper(gym.Wrapper):
    """Wrapper that extracts safety-relevant observations for STL monitoring."""

    def __init__(self, env: gym.Env, max_episode_steps: int = 1000, env_name: str = None):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.observation_space = env.observation_space

        # Auto-detect environment if not specified
        if env_name is None:
            env_name = getattr(env, 'spec', None)
            if env_name is not None:
                env_name = env_name.id
            else:
                env_name = "Ant-v5"  # default

        self.env_name = env_name
        self.config = ENV_CONFIGS.get(env_name, ENV_CONFIGS["Ant-v5"])
        self.torso_id = self.config["torso_id"]
        self.foot_geom_ids = self.config["foot_geom_ids"]

    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs, info = self.env.reset(seed=seed, options=options)
        info["monitor_obs"] = self._extract_monitor_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        info["monitor_obs"] = self._extract_monitor_obs(obs)
        info["step_count"] = self.step_count
        if self.step_count >= self.max_episode_steps:
            truncated = True
        return obs, reward, terminated, truncated, info

    def _extract_monitor_obs(self, obs: np.ndarray) -> dict:
        """Extract safety-relevant signals from MuJoCo simulation state."""
        data = self.env.unwrapped.data
        model = self.env.unwrapped.model
        tid = self.torso_id

        # Orientation from quaternion
        quat = data.xquat[tid]  # w, x, y, z
        roll = self._quat_to_roll(quat)
        pitch = self._quat_to_pitch(quat)

        # Height
        height = float(data.xpos[tid, 2])

        # Velocity (cvel: first 3 = linear, last 3 = angular)
        torso_linvel = data.cvel[tid, :3]

        # External forces (cfrc_ext: first 3 = force, last 3 = torque)
        ext_force = data.cfrc_ext[tid]
        ext_force_norm = float(np.linalg.norm(ext_force[:3]))
        total_ext_force = float(np.sum(np.linalg.norm(data.cfrc_ext[:, :3], axis=1)))

        # Count airborne feet by checking contact geoms
        contacted_feet = set()
        for c_idx in range(data.ncon):
            c = data.contact[c_idx]
            if c.geom1 in self.foot_geom_ids and c.geom2 == 0:
                contacted_feet.add(c.geom1)
            elif c.geom2 in self.foot_geom_ids and c.geom1 == 0:
                contacted_feet.add(c.geom2)
        n_airborne = len(self.foot_geom_ids) - len(contacted_feet)

        # Joints
        joint_offset = self.config["joint_offset"]
        n_joints = self.config["n_joints"]
        vel_offset = self.config["vel_offset"]

        joint_pos = data.qpos[joint_offset:joint_offset+n_joints].copy()
        joint_vel = data.qvel[vel_offset:vel_offset+n_joints].copy()

        # Pad if needed
        if len(joint_pos) < n_joints:
            joint_pos = np.pad(joint_pos, (0, n_joints - len(joint_pos)))
        if len(joint_vel) < n_joints:
            joint_vel = np.pad(joint_vel, (0, n_joints - len(joint_vel)))

        return {
            "body_roll": np.array([roll]),
            "body_pitch": np.array([pitch]),
            "body_height": np.array([height]),
            "body_velocity_x": np.array([float(torso_linvel[0])]),
            "body_velocity_y": np.array([float(torso_linvel[1])]),
            "ext_force_norm": np.array([ext_force_norm]),
            "total_ext_force": np.array([total_ext_force]),
            "n_airborne_feet": np.array([n_airborne], dtype=np.int32),
            "contact_forces": np.array([ext_force_norm] * len(self.foot_geom_ids)),
            "joint_positions": joint_pos,
            "joint_velocities": joint_vel,
        }

    @staticmethod
    def _quat_to_roll(quat: np.ndarray) -> float:
        w, x, y, z = quat
        return float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2)))

    @staticmethod
    def _quat_to_pitch(quat: np.ndarray) -> float:
        w, x, y, z = quat
        sinp = 2.0 * (w * y - z * x)
        return float(np.arcsin(np.clip(sinp, -1.0, 1.0)))
