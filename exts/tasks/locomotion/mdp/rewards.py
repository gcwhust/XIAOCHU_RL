"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import distributions
from typing import TYPE_CHECKING, Optional
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for tracking linear velocity command in x-y plane using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    commands = env.command_manager.get_command(command_name)
    lin_commands = commands[:, :2]
    vel_error = torch.norm(base_lin_vel - lin_commands, dim=1)
    return torch.exp(-vel_error / std**2)

def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for tracking yaw-rate commands about the z-axis via an exponential kernel."""
    asset = env.scene[asset_cfg.name]
    base_ang_vel_z = asset.data.root_ang_vel_w[:, 2]
    commands = env.command_manager.get_command(command_name)
    ang_commands = commands[:, 2]
    ang_error = torch.abs(base_ang_vel_z - ang_commands)
    return torch.exp(-ang_error / std**2)

def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize linear velocity along z-axis using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])

def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize angular velocity along x and y axes using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_w[:, :2]), dim=1)

def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques on the articulation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=2)

def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)

def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)

def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)
    #计算每个关节的功率（力矩乘以角速度），然后对所有关节求和，得到每个环境的总功率惩罚

def pen_base_pitch(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the absolute value of the base pitch angle."""
    asset: RigidObject = env.scene[asset_cfg.name]
    # base rotation is quat (w, x, y, z)
    _, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.abs(pitch)


def knee_joint_pos_l2(
    env: ManagerBasedRLEnv,
    target_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize knee joint angle deviation from target using L2 squared kernel."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - target_angle), dim=1)

def base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.
基于（可选）高度传感器调整目标高度并计算基座 z 与目标的绝对偏差
    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)

def joint_symmetry_l2(
    env: ManagerBasedRLEnv,
    joint_names_a: list[str],
    joint_names_b: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the difference in joint positions between two sets of joints using L2 squared kernel.
    
    This effectively encourages symmetry between the specified joint pairs.
    """
    asset = env.scene[asset_cfg.name]
    # resolve joint indices from names
    # find_joints returns (indices, names)
    ids_a, _ = asset.find_joints(joint_names_a)
    ids_b, _ = asset.find_joints(joint_names_b)
    
    if len(ids_a) != len(ids_b):
        raise ValueError(f"Joint sets must have same length for symmetry reward, got {len(ids_a)} and {len(ids_b)}")

    return torch.sum(torch.square(asset.data.joint_pos[:, ids_a] - asset.data.joint_pos[:, ids_b]), dim=1)

def pen_termination(env: ManagerBasedRLEnv,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
                    ) -> torch.Tensor:
    """Penalize termination events."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return env.termination_manager.terminated.float()


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities when the command is small."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    commands = env.command_manager.get_command(command_name)
    
    # Check if commands are small
    lin_vel_cmd = commands[:, :2] # xy
    ang_vel_cmd = commands[:, 2]  # z
    
    is_still = (torch.norm(lin_vel_cmd, dim=1) < threshold) & (torch.abs(ang_vel_cmd) < threshold)
    
    return torch.sum(torch.square(asset.data.joint_vel), dim=1) * is_still.float()


class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combine rewards
        total_reward = force_reward + velocity_reward
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets >= 0) & (offsets <= 1)), "Offsets must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [gait_indices.view(self.num_envs, 1), (gait_indices + offsets + 1).view(self.num_envs, 1)],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale


class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty
