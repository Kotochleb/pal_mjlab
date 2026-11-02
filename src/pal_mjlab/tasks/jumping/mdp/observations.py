from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg

from .commands import JumpHeightCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def jump_frame_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """World-frame z of the jump command's frame, which targets and apex use."""
  term: JumpHeightCommand = env.command_manager.get_term(command_name)
  return term.height.unsqueeze(-1)


def base_height_world(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """Base-link z in the world frame (ground plane at z = 0)."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]
