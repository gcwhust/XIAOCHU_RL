"""Termination helpers specific to LimX locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Signal termination when the episode length reaches the configured horizon."""

    if not hasattr(env, "episode_length_buf"):
        raise AttributeError("Environment is missing 'episode_length_buf', required for time_out checks.")
    if not hasattr(env, "max_episode_length"):
        raise AttributeError("Environment is missing 'max_episode_length', required for time_out checks.")

    return env.episode_length_buf >= env.max_episode_length

def low_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.4,
) -> torch.Tensor:
    """Terminate if the articulated base height drops below the given threshold."""

    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return base_height < height_threshold