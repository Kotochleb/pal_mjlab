"""Vertical-jump rewards; see arXiv:2510.24584, Table IV and Olympus Lab."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch
from mjlab.managers import RewardTermCfg, SceneEntityCfg

from .commands import FLIGHT, JumpHeightCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _jump(env: ManagerBasedRlEnv, command_name: str) -> JumpHeightCommand:
  term = env.command_manager.get_term(command_name)
  term.update_state()
  return term


def height_kernel(
  error: torch.Tensor, std: float = 0.1, laplace_scale: float = 0.05
) -> torch.Tensor:
  """Broad Gaussian plus a sharper Laplacian, as in the reference code."""
  return torch.exp(-(error / std).square()) + 3 * torch.exp(
    -error.abs() / laplace_scale
  )


def jump_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.1,
  laplace_scale: float = 0.05,
) -> torch.Tensor:
  term = _jump(env, command_name)
  valid = (
    term.just_apex
    & ~term.initialized_in_flight
    & (term.air_time >= term.cfg.min_air_time)
    & (term.peak_height >= term.cfg.min_jump_height)
  )
  reward = height_kernel(term.peak_height - term.target_height, std, laplace_scale)
  # This is an impulse, not a reward rate. Match the paper's once-per-apex
  # payout despite the reward manager's default multiplication by step_dt.
  scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
  return reward * valid / scale


def estimated_jump_height(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.1,
  laplace_scale: float = 0.05,
) -> torch.Tensor:
  term = _jump(env, command_name)
  vz = term.robot.data.root_link_lin_vel_w[:, 2]
  estimated = term.height + vz.clamp_min(0).square() / (2 * term.gravity)
  ascending = (term.phase == FLIGHT) & ~term.apex_reached & (vz > 0)
  return height_kernel(estimated - term.target_height, std, laplace_scale) * ascending


class height_variable_posture:
  """exp(-mean((q-q_default)^2/std(height)^2)), with per-joint tolerances.

  Interpolate ground/airborne stds using measured root rise, clamped at both
  ends. This preserves the robot's own default configuration and lets it crouch
  and extend near the ground. Targets and passive linkage joints are unchanged.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    robot = env.scene[asset_cfg.name]
    self.joint_ids = asset_cfg.joint_ids
    names = (
      robot.joint_names[self.joint_ids]
      if isinstance(self.joint_ids, slice)
      else [robot.joint_names[i] for i in self.joint_ids]
    )
    if not names:
      raise ValueError("Height posture reward requires at least one joint.")
    self.stds = []
    for key in ("std_ground", "std_airborne"):
      mapping = cfg.params[key]
      values = []
      for name in names:
        matches = [
          value for pattern, value in mapping.items() if re.fullmatch(pattern, name)
        ]
        if len(matches) != 1 or not math.isfinite(matches[0]) or matches[0] <= 0:
          raise ValueError(
            f"{key} must match {name!r} once with a positive finite std."
          )
        values.append(matches[0])
      self.stds.append(torch.tensor(values, device=env.device))
    low, high = cfg.params["height_range"]
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
      raise ValueError("Posture height_range must be finite and strictly increasing.")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std_ground: dict[str, float],
    std_airborne: dict[str, float],
    height_range: tuple[float, float],
  ) -> torch.Tensor:
    del std_ground, std_airborne
    term = _jump(env, command_name)
    low, high = height_range
    blend = ((term.height - low) / (high - low)).clamp(0, 1).unsqueeze(-1)
    std = torch.lerp(self.stds[0], self.stds[1], blend)
    robot = env.scene[asset_cfg.name]
    error = (
      robot.data.joint_pos[:, self.joint_ids]
      - robot.data.default_joint_pos[:, self.joint_ids]
    )
    return torch.exp(-(error / std).square().mean(dim=-1))


def horizontal_velocity(
  env: ManagerBasedRlEnv, command_name: str, std: float = 0.2
) -> torch.Tensor:
  """Reference implementation's horizontal-motion suppression for vertical jumps."""
  term = _jump(env, command_name)
  speed = term.robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
  return torch.exp(-speed / std**2)


