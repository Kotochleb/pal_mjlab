"""Vertical-jump rewards; see arXiv:2510.24584, Table IV and Olympus Lab."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch
from mjlab.managers import RewardTermCfg, SceneEntityCfg

from .commands import FLIGHT, STANCE, JumpHeightCommand, match_joint_values
from .terminations import jump_finished

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
    & (term.air_time >= term.cfg.min_air_time)
    & (term.peak_height >= term.cfg.min_jump_height)
  )
  reward = height_kernel(term.peak_height - term.target_height, std, laplace_scale)
  # This is an impulse, not a reward rate. Match the paper's once-per-apex
  # payout despite the reward manager's default multiplication by step_dt.
  scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
  return reward * valid / scale


def landing_completed(
  env: ManagerBasedRlEnv, command_name: str, hold_time: float | None = None
) -> torch.Tensor:
  """exp(-mean((q-q_default)^2/std^2)) once, on the jump_finished step.

  Sparse: paid only on the step the robot has held SETTLED for hold_time
  (default: the command's settle_hold_time), the step the jump_finished
  termination ends the episode. The payout grows to 1
  at the reference (exp(-1) at an RMS error of one settle_pose_std); zero on
  every other step.
  """
  term = _jump(env, command_name)
  done = jump_finished(env, command_name, hold_time)
  # An impulse, not a rate: undo the reward manager's multiplication by step_dt.
  scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
  return torch.exp(-term.pose_error) * done / scale


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

  Interpolate ground/airborne stds using the measured world-frame base height,
  clamped at both ends of height_range (world z). This preserves the robot's own default configuration and lets it crouch
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
    self.stds = [
      torch.tensor(match_joint_values(cfg.params[key], names, key), device=env.device)
      for key in ("std_ground", "std_airborne")
    ]
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


class joint_symmetry_l2:
  """Sum over left/right joint pairs of (q_left - s * q_right)^2.

  Pairs are built once by swapping "_left_" for "_right_" in each selected left
  joint name. s = -1 for joints matching any `mirrored_joints` pattern (equal
  and opposite, e.g. yaw/roll with identical left/right axes), else +1 (equal).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    robot = env.scene[asset_cfg.name]
    joint_names = list(robot.joint_names)
    ids = asset_cfg.joint_ids
    selected = (
      joint_names[ids] if isinstance(ids, slice) else [joint_names[i] for i in ids]
    )
    left_ids, right_ids, mirrored = [], [], []
    for name in selected:
      if "_left_" not in name:
        continue
      partner = name.replace("_left_", "_right_")
      if partner not in selected:
        raise ValueError(f"Symmetry partner {partner!r} of {name!r} is not selected.")
      left_ids.append(joint_names.index(name))
      right_ids.append(joint_names.index(partner))
      mirrored.append(any(re.fullmatch(p, name) for p in cfg.params["mirrored_joints"]))
    if not left_ids or 2 * len(left_ids) != len(selected):
      raise ValueError("Symmetry cost requires complete left/right joint pairs.")
    self.left_ids = torch.tensor(left_ids, device=env.device)
    self.right_ids = torch.tensor(right_ids, device=env.device)
    self.mirrored = torch.tensor(mirrored, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    mirrored_joints: tuple[str, ...],
  ) -> torch.Tensor:
    del mirrored_joints
    joint_pos = env.scene[asset_cfg.name].data.joint_pos
    left = joint_pos[:, self.left_ids]
    right = joint_pos[:, self.right_ids]
    target = torch.where(self.mirrored, -right, right)
    return (left - target).square().sum(dim=-1)


def horizontal_velocity(
  env: ManagerBasedRlEnv, command_name: str, std: float = 0.2
) -> torch.Tensor:
  """Reference implementation's horizontal-motion suppression for vertical jumps."""
  term = _jump(env, command_name)
  speed = term.robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
  return torch.exp(-speed / std**2)


def ground_force_l2(
  env: ManagerBasedRlEnv, sensor_name: str, force_scale: float = 500.0
) -> torch.Tensor:
  """Sum of squared foot-ground contact forces, normalized by force_scale."""
  force = env.scene[sensor_name].data.force
  return (force.norm(dim=-1) / force_scale).square().sum(dim=-1)


def feet_slip(
  env: ManagerBasedRlEnv, sensor_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Sum over feet in ground contact of their squared horizontal site speed.

  mjlab's velocity feet_slip without its twist-command gate: active in every
  phase, so stance, takeoff push-off and landing all penalize sliding feet.
  """
  in_contact = (env.scene[sensor_name].data.found > 0).float()
  speed = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  speed = speed.norm(dim=-1)
  env.extras["log"]["Metrics/slip_velocity_mean"] = (speed * in_contact).sum() / (
    in_contact.sum().clamp_min(1)
  )
  return (speed.square() * in_contact).sum(dim=-1)


def landing_drift_l2(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Squared horizontal COM distance from its takeoff position, after takeoff.

  Active in flight and after touchdown, so the policy must land, and settle,
  where it took off. Tilt is not penalized; only where the COM ends up is.
  """
  term = _jump(env, command_name)
  airborne_or_landed = term.phase != STANCE
  drift = (term.com_xy - term.takeoff_com_xy).square().sum(dim=-1)
  return drift * airborne_or_landed
